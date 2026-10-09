import pytest

from rag.chunking import chunk_text


def test_chunk_text_with_overlap() -> None:
    chunks = chunk_text(
        "abcdefghij",
        size=6,
        overlap=2,
    )

    assert chunks == [
        "abcdef",
        "efghij",
        "ij",
    ]


def test_chunk_text_returns_empty_list() -> None:
    assert chunk_text("") == []


def test_chunk_text_accepts_zero_overlap() -> None:
    assert chunk_text(
        "abcdefgh",
        size=4,
        overlap=0,
    ) == ["abcd", "efgh"]


@pytest.mark.parametrize(
    ("size", "overlap"),
    [
        (0, 0),
        (-1, 0),
        (5, -1),
        (5, 5),
        (5, 6),
    ],
)
def test_chunk_text_rejects_invalid_parameters(
    size: int,
    overlap: int,
) -> None:
    with pytest.raises(ValueError):
        chunk_text(
            "hello",
            size=size,
            overlap=overlap,
        )
