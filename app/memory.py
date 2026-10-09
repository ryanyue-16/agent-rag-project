from __future__ import annotations

from dataclasses import dataclass

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
)


@dataclass(frozen=True)
class ConversationTurn:
    """保存一轮已经完成的用户问题和助手回答。"""

    user: str
    assistant: str


class ConversationMemory:
    """
    保存当前会话中的多轮对话。

    只保留最近 max_turns 轮，避免历史无限增长，
    导致 Prompt 越来越长和 Token 消耗持续增加。
    """

    def __init__(
        self,
        max_turns: int = 6,
    ) -> None:
        if max_turns <= 0:
            raise ValueError(
                "max_turns 必须大于 0"
            )

        self.max_turns = max_turns
        self._turns: list[
            ConversationTurn
        ] = []

    def __len__(self) -> int:
        """返回当前保存的对话轮数。"""
        return len(self._turns)

    def add_turn(
        self,
        user: str,
        assistant: str,
    ) -> None:
        """保存一轮成功完成的对话。"""
        clean_user = user.strip()
        clean_assistant = assistant.strip()

        if not clean_user:
            raise ValueError(
                "用户问题不能为空"
            )

        if not clean_assistant:
            raise ValueError(
                "助手回答不能为空"
            )

        self._turns.append(
            ConversationTurn(
                user=clean_user,
                assistant=clean_assistant,
            )
        )

        if len(self._turns) > self.max_turns:
            self._turns = self._turns[
                -self.max_turns:
            ]

    def format_history(self) -> str:
        """
        将历史格式化为便于阅读的字符串。

        当前 Tool Calling Agent 实际使用的是 to_messages()；
        本方法保留用于调试、展示和测试。
        """
        if not self._turns:
            return "（暂无历史对话）"

        blocks: list[str] = []

        for index, turn in enumerate(
            self._turns,
            start=1,
        ):
            blocks.append(
                "\n".join(
                    [
                        f"第 {index} 轮",
                        f"用户：{turn.user}",
                        f"助手：{turn.assistant}",
                    ]
                )
            )

        return "\n\n".join(blocks)

    def to_messages(
        self,
    ) -> list[BaseMessage]:
        """
        将历史转换为 LangChain Message 对象。

        Tool Calling Agent 接收真正的 HumanMessage
        和 AIMessage，而不是手工拼接的大段字符串。
        """
        messages: list[BaseMessage] = []

        for turn in self._turns:
            messages.append(
                HumanMessage(
                    content=turn.user,
                )
            )

            messages.append(
                AIMessage(
                    content=turn.assistant,
                )
            )

        return messages

    def clear(self) -> None:
        """清空当前会话的全部历史。"""
        self._turns.clear()
