from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from threading import RLock
from typing import Any, Callable

from pypdf import PdfReader

from rag.chunking import build_chunks
from rag.qa_system import RAGQASystem
from rag.retrievers import BM25Retriever, CrossEncoderReranker

COLLECTION_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
MetadataFilter = dict[str, str | list[str]]
PageLoader = Callable[[bytes], list[str]]


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def extract_pdf_pages(content: bytes) -> list[str]:
    """Extract non-empty text pages from a text-based PDF."""
    if not content.startswith(b"%PDF"):
        raise ValueError("上传内容不是有效 PDF")
    try:
        reader = PdfReader(BytesIO(content))
    except Exception as exc:
        raise ValueError("无法解析 PDF") from exc
    pages = [" ".join((page.extract_text() or "").split()) for page in reader.pages]
    if not any(pages):
        raise ValueError("PDF 不包含可提取文本；OCR 暂不支持")
    return pages


class KnowledgeBaseService:
    """Versioned persistent collection store with atomic active-index swaps."""

    def __init__(
        self,
        *,
        root: str | Path,
        client: Any,
        embed_model: str,
        chat_model: str,
        retrieval_mode: str,
        candidate_k: int,
        rrf_k: int,
        reranker_model: str,
        reranker: CrossEncoderReranker | None = None,
        page_loader: PageLoader = extract_pdf_pages,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.client = client
        self.embed_model = embed_model
        self.chat_model = chat_model
        self.retrieval_mode = retrieval_mode
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.reranker_model = reranker_model
        self.reranker = reranker
        self.page_loader = page_loader
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._lock = RLock()
        self._active: dict[str, tuple[str, RAGQASystem | None]] = {}

    @staticmethod
    def _validate_collection_id(collection_id: str) -> str:
        clean = collection_id.strip().lower()
        if not COLLECTION_ID_PATTERN.fullmatch(clean):
            raise ValueError(
                "collection_id 只能包含小写字母、数字、下划线和连字符，最长 64 字符"
            )
        return clean

    def _collection_dir(self, collection_id: str) -> Path:
        return self.root / self._validate_collection_id(collection_id)

    def _require_collection(self, collection_id: str) -> tuple[str, Path]:
        clean = self._validate_collection_id(collection_id)
        directory = self.root / clean
        if not (directory / "collection.json").exists():
            raise KeyError(f"collection 不存在：{clean}")
        return clean, directory

    @staticmethod
    def _public_document(record: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in record.items() if key != "stored_name"}

    def create_collection(
        self,
        collection_id: str,
        *,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        clean = self._validate_collection_id(collection_id)
        if not name.strip():
            raise ValueError("collection name 不能为空")
        directory = self.root / clean
        with self._lock:
            if directory.exists():
                raise FileExistsError(f"collection 已存在：{clean}")
            (directory / "documents").mkdir(parents=True)
            (directory / "versions").mkdir()
            payload = {
                "collection_id": clean,
                "name": name.strip(),
                "metadata": metadata or {},
                "created_at": _now(),
            }
            _atomic_json(directory / "collection.json", payload)
            _atomic_json(directory / "documents.json", {"documents": []})
            self._build_and_activate(clean, directory, [])
            return payload

    def list_collections(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        with self._lock:
            for path in sorted(self.root.iterdir()):
                metadata_path = path / "collection.json"
                if not path.is_dir() or not metadata_path.exists():
                    continue
                payload = _read_json(metadata_path)
                documents = self._documents(path)
                payload["document_count"] = len(documents)
                payload["active_version"] = self._current_version(path)
                results.append(payload)
        return results

    def get_collection(self, collection_id: str) -> dict[str, Any]:
        clean, directory = self._require_collection(collection_id)
        payload = _read_json(directory / "collection.json")
        documents = self._documents(directory)
        payload.update(
            {
                "collection_id": clean,
                "document_count": len(documents),
                "active_version": self._current_version(directory),
            }
        )
        return payload

    @staticmethod
    def _documents(directory: Path) -> list[dict[str, Any]]:
        payload = _read_json(directory / "documents.json", {"documents": []})
        return list(payload.get("documents", []))

    def list_documents(self, collection_id: str) -> list[dict[str, Any]]:
        _, directory = self._require_collection(collection_id)
        return [self._public_document(item) for item in self._documents(directory)]

    def get_document(
        self, collection_id: str, document_id: str
    ) -> dict[str, Any]:
        _, directory = self._require_collection(collection_id)
        record = next(
            (
                item
                for item in self._documents(directory)
                if item["document_id"] == document_id
            ),
            None,
        )
        if record is None:
            raise KeyError(f"document 不存在：{document_id}")
        return self._public_document(record)

    def upload_document(
        self,
        collection_id: str,
        *,
        filename: str,
        content: bytes,
        title: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        clean, directory = self._require_collection(collection_id)
        if not filename.lower().endswith(".pdf"):
            raise ValueError("仅支持 .pdf 文件")
        pages = self.page_loader(content)
        digest = hashlib.sha256(content).hexdigest()
        with self._lock:
            previous = self._documents(directory)
            if any(item["sha256"] == digest for item in previous):
                raise FileExistsError("该 PDF 已存在于 collection 中")
            document_id = uuid.uuid4().hex
            stored_name = f"{document_id}.pdf"
            record = {
                "document_id": document_id,
                "filename": Path(filename).name,
                "title": (title or Path(filename).stem).strip(),
                "tags": sorted({tag.strip() for tag in tags or [] if tag.strip()}),
                "metadata": metadata or {},
                "sha256": digest,
                "page_count": len(pages),
                "created_at": _now(),
                "stored_name": stored_name,
            }
            document_path = directory / "documents" / stored_name
            _atomic_bytes(document_path, content)
            updated = [*previous, record]
            _atomic_json(directory / "documents.json", {"documents": updated})
            try:
                self._build_and_activate(clean, directory, updated)
            except Exception:
                _atomic_json(directory / "documents.json", {"documents": previous})
                document_path.unlink(missing_ok=True)
                raise
            return self._public_document(record)

    def delete_document(self, collection_id: str, document_id: str) -> bool:
        clean, directory = self._require_collection(collection_id)
        with self._lock:
            previous = self._documents(directory)
            target = next(
                (item for item in previous if item["document_id"] == document_id),
                None,
            )
            if target is None:
                return False
            updated = [
                item for item in previous if item["document_id"] != document_id
            ]
            _atomic_json(directory / "documents.json", {"documents": updated})
            try:
                self._build_and_activate(clean, directory, updated)
            except Exception:
                _atomic_json(directory / "documents.json", {"documents": previous})
                raise
            (directory / "documents" / target["stored_name"]).unlink(missing_ok=True)
            return True

    def rebuild(self, collection_id: str) -> dict[str, Any]:
        clean, directory = self._require_collection(collection_id)
        with self._lock:
            return self._build_and_activate(
                clean, directory, self._documents(directory)
            )

    def import_directory(self, collection_id: str, pdf_dir: str | Path) -> int:
        """Bootstrap an empty collection from local PDFs with one rebuild."""
        clean, directory = self._require_collection(collection_id)
        paths = sorted(Path(pdf_dir).glob("*.pdf"))
        with self._lock:
            if self._documents(directory):
                return 0
            records: list[dict[str, Any]] = []
            for path in paths:
                content = path.read_bytes()
                pages = self.page_loader(content)
                document_id = uuid.uuid4().hex
                stored_name = f"{document_id}.pdf"
                _atomic_bytes(directory / "documents" / stored_name, content)
                records.append(
                    {
                        "document_id": document_id,
                        "filename": path.name,
                        "title": path.stem,
                        "tags": [],
                        "metadata": {},
                        "sha256": hashlib.sha256(content).hexdigest(),
                        "page_count": len(pages),
                        "created_at": _now(),
                        "stored_name": stored_name,
                    }
                )
            _atomic_json(directory / "documents.json", {"documents": records})
            try:
                self._build_and_activate(clean, directory, records)
            except Exception:
                _atomic_json(directory / "documents.json", {"documents": []})
                for record in records:
                    (directory / "documents" / record["stored_name"]).unlink(
                        missing_ok=True
                    )
                raise
            return len(records)

    def _document_pages(
        self,
        collection_id: str,
        directory: Path,
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        documents: list[dict[str, Any]] = []
        for record in records:
            content = (directory / "documents" / record["stored_name"]).read_bytes()
            pages = self.page_loader(content)
            for page_number, text in enumerate(pages, start=1):
                if not text:
                    continue
                documents.append(
                    {
                        "text": text,
                        "page": page_number,
                        "source": record["filename"],
                        "document_id": record["document_id"],
                        "collection_id": collection_id,
                        "metadata": {
                            **record.get("metadata", {}),
                            "title": record["title"],
                            "tags": record["tags"],
                        },
                    }
                )
        return documents

    def _build_and_activate(
        self,
        collection_id: str,
        directory: Path,
        records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        pages = self._document_pages(collection_id, directory, records)
        chunks = build_chunks(
            pages, size=self.chunk_size, overlap=self.chunk_overlap
        ) if pages else []
        system: RAGQASystem | None = None
        if chunks:
            build_mode = (
                "hybrid" if self.retrieval_mode == "bm25" else self.retrieval_mode
            )
            system = RAGQASystem(
                chunks=chunks,
                client=self.client,
                embed_model=self.embed_model,
                chat_model=self.chat_model,
                retrieval_mode=build_mode,
                candidate_k=self.candidate_k,
                rrf_k=self.rrf_k,
                reranker_model=self.reranker_model,
                reranker=self.reranker,
            )
            system.retrieval_mode = self.retrieval_mode

        version_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        version_id = f"{version_id}-{uuid.uuid4().hex[:8]}"
        versions = directory / "versions"
        temporary = versions / f".tmp-{version_id}"
        final = versions / version_id
        temporary.mkdir(parents=True)
        try:
            _atomic_json(temporary / "chunks.json", {"chunks": chunks})
            if system is not None:
                bm25_snapshot = system.bm25 or BM25Retriever(chunks)
                _atomic_json(temporary / "bm25.json", bm25_snapshot.to_dict())
            if system is not None and system.index is not None:
                import faiss

                faiss.write_index(system.index, str(temporary / "faiss.index"))
            manifest = {
                "version_id": version_id,
                "collection_id": collection_id,
                "created_at": _now(),
                "document_count": len(records),
                "chunk_count": len(chunks),
                "retrieval_mode": self.retrieval_mode,
                "embed_model": self.embed_model,
                "bm25_persisted": (temporary / "bm25.json").exists(),
                "faiss_persisted": (temporary / "faiss.index").exists(),
            }
            _atomic_json(temporary / "manifest.json", manifest)
            os.replace(temporary, final)
            _atomic_text(directory / "CURRENT", version_id + "\n")
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        self._active[collection_id] = (version_id, system)
        return manifest

    @staticmethod
    def _current_version(directory: Path) -> str | None:
        pointer = directory / "CURRENT"
        return pointer.read_text(encoding="utf-8").strip() if pointer.exists() else None

    def _load_active(self, collection_id: str) -> RAGQASystem | None:
        clean, directory = self._require_collection(collection_id)
        version_id = self._current_version(directory)
        if version_id is None:
            return None
        cached = self._active.get(clean)
        if cached is not None and cached[0] == version_id:
            return cached[1]
        version = directory / "versions" / version_id
        manifest = _read_json(version / "manifest.json", {})
        if manifest.get("embed_model") != self.embed_model:
            raise RuntimeError(
                "持久化索引的 embedding model 与当前配置不一致，请重建索引"
            )
        chunks = _read_json(version / "chunks.json", {"chunks": []})["chunks"]
        if not chunks:
            self._active[clean] = (version_id, None)
            return None
        dense_index = None
        if (version / "faiss.index").exists():
            import faiss

            dense_index = faiss.read_index(str(version / "faiss.index"))
        bm25 = None
        if (version / "bm25.json").exists():
            bm25 = BM25Retriever.from_dict(
                chunks, _read_json(version / "bm25.json")
            )
        system = RAGQASystem(
            chunks=chunks,
            client=self.client,
            embed_model=self.embed_model,
            chat_model=self.chat_model,
            retrieval_mode=self.retrieval_mode,
            candidate_k=self.candidate_k,
            rrf_k=self.rrf_k,
            reranker_model=self.reranker_model,
            reranker=self.reranker,
            dense_index=dense_index,
            bm25_retriever=bm25,
        )
        self._active[clean] = (version_id, system)
        return system

    def retrieve_scoped(
        self,
        query: str,
        *,
        top_k: int,
        collection_id: str,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve only from the explicitly authorized collection."""
        with self._lock:
            system = self._load_active(collection_id)
        if system is None:
            return []
        return system.retrieve(
            query, top_k=top_k, metadata_filter=metadata_filter
        )

    def document_names(self, collection_id: str) -> list[str]:
        return [item["filename"] for item in self.list_documents(collection_id)]

    def is_ready(self) -> bool:
        """Check that every registered collection points to a valid manifest."""
        collections = self.list_collections()
        if not collections:
            return False
        for collection in collections:
            version_id = collection.get("active_version")
            if not version_id:
                return False
            manifest = (
                self.root
                / collection["collection_id"]
                / "versions"
                / str(version_id)
                / "manifest.json"
            )
            if not manifest.exists():
                return False
        return True
