from __future__ import annotations

import re

SEPARATOR_PATTERN = re.compile(r"[,，;；]")


def decompose_query(query: str) -> list[str]:
    """Return the full query plus deterministic multi-hop clauses."""
    normalized = " ".join(query.split()).strip()
    if not normalized:
        return []

    variants = [normalized]
    stripped = normalized.rstrip("?？.!。 ")
    clauses = [
        clause.strip()
        for clause in SEPARATOR_PATTERN.split(stripped)
        if len(clause.strip()) >= 4
    ]
    if len(clauses) > 1:
        variants.extend(clauses)

    if "才能" in stripped:
        before, after = stripped.split("才能", maxsplit=1)
        before = before.strip()
        after = after.strip()
        if len(before) >= 4 and len(after) >= 4:
            variants.extend((before, after))

    # “A 和 B 分别……” explicitly asks for two independent facts.
    respectively = re.match(
        r"^(?P<left>.+?)和(?P<right>.+?)分别(?P<tail>.+)$",
        stripped,
    )
    if respectively:
        tail = respectively.group("tail").strip()
        variants.extend(
            (
                f"{respectively.group('left').strip()}{tail}",
                f"{respectively.group('right').strip()}{tail}",
            )
        )

    deduplicated: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        key = variant.casefold()
        if key not in seen:
            seen.add(key)
            deduplicated.append(variant)
    return deduplicated[:3]
