from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.server import create_app
from api.session_store import SessionStore
from rag.knowledge_base import KnowledgeBaseService


class FakeEmbeddings:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.fail = False

    def create(self, *, model: str, input: list[str]):
        if self.fail:
            raise RuntimeError("embedding failure")
        self.calls.append(list(input))
        data = []
        for text in input:
            lowered = text.casefold()
            vector = np.array(
                [
                    float(lowered.count("alpha")),
                    float(lowered.count("beta")),
                    0.1,
                ],
                dtype="float32",
            )
            data.append(SimpleNamespace(embedding=vector.tolist()))
        return SimpleNamespace(data=data)


def make_service(tmp_path, embeddings: FakeEmbeddings) -> KnowledgeBaseService:
    return KnowledgeBaseService(
        root=tmp_path / "indexes",
        client=SimpleNamespace(embeddings=embeddings),
        embed_model="fake-embedding",
        chat_model="unused",
        retrieval_mode="hybrid",
        candidate_k=20,
        rrf_k=60,
        reranker_model="unused",
        page_loader=lambda content: [content.decode("utf-8")],
        chunk_size=500,
        chunk_overlap=0,
    )


def upload(
    service: KnowledgeBaseService,
    collection_id: str,
    filename: str,
    text: str,
    tag: str,
) -> dict[str, Any]:
    return service.upload_document(
        collection_id,
        filename=filename,
        content=text.encode(),
        tags=[tag],
        metadata={"department": tag},
    )


def test_persistent_indexes_scope_filter_and_delete(tmp_path) -> None:
    embeddings = FakeEmbeddings()
    service = make_service(tmp_path, embeddings)
    service.create_collection("team-a", name="Team A")
    service.create_collection("team-b", name="Team B")
    alpha = upload(service, "team-a", "alpha.pdf", "alpha policy", "hr")
    beta = upload(service, "team-a", "beta.pdf", "beta policy", "security")
    secret = upload(service, "team-b", "secret.pdf", "alpha secret", "private")

    team_a = service.retrieve_scoped(
        "alpha", top_k=3, collection_id="team-a"
    )
    assert team_a[0]["document_id"] == alpha["document_id"]
    assert secret["document_id"] not in {item["document_id"] for item in team_a}

    filtered = service.retrieve_scoped(
        "policy",
        top_k=3,
        collection_id="team-a",
        metadata_filter={"tags": ["security"]},
    )
    assert {item["document_id"] for item in filtered} == {beta["document_id"]}

    collection_dir = tmp_path / "indexes" / "team-a"
    version = (collection_dir / "CURRENT").read_text().strip()
    version_dir = collection_dir / "versions" / version
    assert (version_dir / "faiss.index").exists()
    assert (version_dir / "bm25.json").exists()
    assert (version_dir / "chunks.json").exists()
    assert (version_dir / "manifest.json").exists()

    # A fresh service restores document indexes rather than embedding them again.
    before_restart = len(embeddings.calls)
    restored = make_service(tmp_path, embeddings)
    restored_results = restored.retrieve_scoped(
        "alpha", top_k=3, collection_id="team-a"
    )
    assert restored_results[0]["document_id"] == alpha["document_id"]
    assert len(embeddings.calls) == before_restart + 1

    old_version = version
    assert restored.delete_document("team-a", alpha["document_id"]) is True
    new_version = (collection_dir / "CURRENT").read_text().strip()
    assert new_version != old_version
    after_delete = restored.retrieve_scoped(
        "alpha", top_k=5, collection_id="team-a"
    )
    assert alpha["document_id"] not in {
        item["document_id"] for item in after_delete
    }
    assert not any(
        item["document_id"] == alpha["document_id"]
        for item in restored.list_documents("team-a")
    )


def test_failed_rebuild_keeps_active_version_and_rolls_back(tmp_path) -> None:
    embeddings = FakeEmbeddings()
    service = make_service(tmp_path, embeddings)
    service.create_collection("stable", name="Stable")
    upload(service, "stable", "alpha.pdf", "alpha policy", "hr")
    current_path = tmp_path / "indexes" / "stable" / "CURRENT"
    stable_version = current_path.read_text().strip()

    embeddings.fail = True
    with pytest.raises(RuntimeError, match="embedding failure"):
        upload(service, "stable", "beta.pdf", "beta policy", "security")

    assert current_path.read_text().strip() == stable_version
    assert [item["filename"] for item in service.list_documents("stable")] == [
        "alpha.pdf"
    ]


def test_collection_document_api(tmp_path) -> None:
    embeddings = FakeEmbeddings()
    service = make_service(tmp_path, embeddings)

    class Workflow:
        knowledge_base = service
        supports_thread_state = False

        def invoke(self, inputs):
            return {"output": "unused"}

    app = create_app(
        workflow=Workflow(), session_store=SessionStore(max_turns=2)
    )
    with TestClient(app) as client:
        created = client.post(
            "/collections",
            json={"collection_id": "docs", "name": "Documents"},
        )
        uploaded = client.post(
            "/collections/docs/documents",
            files={"file": ("alpha.pdf", b"alpha handbook", "application/pdf")},
            data={"tags": "hr,policy", "metadata": '{"region":"uk"}'},
        )
        listed = client.get("/collections/docs/documents")
        rebuilt = client.post("/collections/docs/rebuild")
        document_id = uploaded.json()["document_id"]
        viewed = client.get(f"/collections/docs/documents/{document_id}")
        deleted = client.delete(
            f"/collections/docs/documents/{document_id}"
        )

    assert created.status_code == 201
    assert uploaded.status_code == 201
    assert listed.json()[0]["metadata"] == {"region": "uk"}
    assert viewed.json()["document_id"] == document_id
    assert rebuilt.status_code == 200
    assert deleted.json()["deleted"] is True
