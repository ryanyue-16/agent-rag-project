import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)

from app.memory import ConversationMemory


def test_memory_formats_turns_in_order() -> None:
    memory = ConversationMemory(
        max_turns=3
    )

    memory.add_turn(
        user="第一问",
        assistant="第一答",
    )
    memory.add_turn(
        user="第二问",
        assistant="第二答",
    )

    history = memory.format_history()

    assert "用户：第一问" in history
    assert "助手：第一答" in history
    assert "用户：第二问" in history
    assert "助手：第二答" in history

    assert history.index("第一问") < (
        history.index("第二问")
    )


def test_memory_keeps_only_recent_turns() -> None:
    memory = ConversationMemory(
        max_turns=2
    )

    memory.add_turn(
        "问题1",
        "回答1",
    )
    memory.add_turn(
        "问题2",
        "回答2",
    )
    memory.add_turn(
        "问题3",
        "回答3",
    )

    history = memory.format_history()

    assert len(memory) == 2
    assert "问题1" not in history
    assert "问题2" in history
    assert "问题3" in history


def test_memory_can_be_cleared() -> None:
    memory = ConversationMemory()

    memory.add_turn(
        "什么是 RAG？",
        "RAG 是检索增强生成。",
    )

    memory.clear()

    assert len(memory) == 0
    assert memory.format_history() == (
        "（暂无历史对话）"
    )
    assert memory.to_messages() == []


def test_memory_rejects_invalid_max_turns() -> None:
    with pytest.raises(
        ValueError,
        match="max_turns 必须大于 0",
    ):
        ConversationMemory(
            max_turns=0
        )


def test_memory_converts_turns_to_messages() -> None:
    memory = ConversationMemory(
        max_turns=3
    )

    memory.add_turn(
        user="候选人叫什么？",
        assistant=(
            "候选人叫 Yuhao Yue。"
        ),
    )

    messages = memory.to_messages()

    assert len(messages) == 2

    assert isinstance(
        messages[0],
        HumanMessage,
    )
    assert messages[0].content == (
        "候选人叫什么？"
    )

    assert isinstance(
        messages[1],
        AIMessage,
    )
    assert messages[1].content == (
        "候选人叫 Yuhao Yue。"
    )
