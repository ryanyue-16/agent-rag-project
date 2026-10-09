import json

import pytest

from evals.corpus import load_eval_chunks


def test_load_eval_chunks(tmp_path):
    path = tmp_path / "corpus.json"
    path.write_text(
        json.dumps(
            [
                {
                    "chunk_id": "policy-p1-c0",
                    "source": "policy.pdf",
                    "page": 1,
                    "text": "Access must be revoked within 24 hours.",
                }
            ]
        ),
        encoding="utf-8",
    )

    chunks = load_eval_chunks(path)

    assert chunks[0]["chunk_id"] == "policy-p1-c0"


def test_duplicate_chunk_id_is_rejected(tmp_path):
    path = tmp_path / "corpus.json"
    chunk = {
        "chunk_id": "same",
        "source": "policy.pdf",
        "page": 1,
        "text": "text",
    }
    path.write_text(
        json.dumps([chunk, chunk]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="chunk_id 重复"):
        load_eval_chunks(path)
