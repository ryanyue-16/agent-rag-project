from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.application import build_application
from app.config import DEFAULT_COLLECTION_ID, INDEX_DIR


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a real persisted-index Stage 5 smoke evaluation."
    )
    parser.add_argument(
        "--output",
        default="evals/reports/stage5_persistence_smoke.json",
    )
    args = parser.parse_args()

    # Building twice verifies that the second application can load the active
    # version left by the first process-level construction.
    first = build_application()
    first_collection = first.knowledge_base.get_collection(
        DEFAULT_COLLECTION_ID
    )
    second = build_application()
    service = second.knowledge_base
    collection = service.get_collection(DEFAULT_COLLECTION_ID)
    results = service.retrieve_scoped(
        "Python 入门指南主要介绍什么？",
        top_k=3,
        collection_id=DEFAULT_COLLECTION_ID,
    )

    version_id = collection["active_version"]
    version_dir = Path(INDEX_DIR) / DEFAULT_COLLECTION_ID / "versions" / version_id
    checks = {
        "same_version_after_restart": (
            first_collection["active_version"] == version_id
        ),
        "documents_present": collection["document_count"] > 0,
        "faiss_persisted": (version_dir / "faiss.index").exists(),
        "bm25_persisted": (version_dir / "bm25.json").exists(),
        "chunks_persisted": (version_dir / "chunks.json").exists(),
        "manifest_persisted": (version_dir / "manifest.json").exists(),
        "real_retrieval_non_empty": bool(results),
    }
    payload = {
        "stage": 5,
        "collection_id": DEFAULT_COLLECTION_ID,
        "document_count": collection["document_count"],
        "retrieved_count": len(results),
        "checks": checks,
        "passed": all(checks.values()),
        "notes": [
            "No raw chunk text or generated answer is stored.",
            "Deletion, filtering and atomic rollback are covered by deterministic tests.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
