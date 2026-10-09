from __future__ import annotations

from threading import RLock

from app.memory import ConversationMemory


class SessionState:
    """
    保存一个 API 会话的 Memory 和锁。

    同一个 session_id 的请求按顺序处理，
    避免并发修改同一份历史。
    """

    def __init__(
        self,
        max_turns: int,
    ) -> None:
        self.memory = ConversationMemory(
            max_turns=max_turns,
        )
        self.lock = RLock()
        self.thread_ids: set[str] = set()


class SessionStore:
    """进程内 API 会话存储。"""

    def __init__(
        self,
        max_turns: int,
    ) -> None:
        if max_turns <= 0:
            raise ValueError(
                "max_turns 必须大于 0"
            )

        self._max_turns = max_turns
        self._sessions: dict[
            str,
            SessionState,
        ] = {}
        self._lock = RLock()

    def __len__(self) -> int:
        """返回当前 Session 数量。"""
        with self._lock:
            return len(self._sessions)

    def get_or_create(
        self,
        session_id: str,
    ) -> SessionState:
        """获取会话；不存在时创建。"""
        clean_session_id = (
            session_id.strip()
        )

        if not clean_session_id:
            raise ValueError(
                "session_id 不能为空"
            )

        with self._lock:
            state = self._sessions.get(
                clean_session_id
            )

            if state is None:
                state = SessionState(
                    max_turns=(
                        self._max_turns
                    ),
                )

                self._sessions[
                    clean_session_id
                ] = state

            return state

    def clear(
        self,
        session_id: str,
    ) -> bool:
        """
        清空指定会话。

        True：会话存在并已清空。
        False：会话不存在。
        """
        clean_session_id = (
            session_id.strip()
        )

        if not clean_session_id:
            return False

        with self._lock:
            state = self._sessions.get(
                clean_session_id
            )

        if state is None:
            return False

        with state.lock:
            state.memory.clear()
            state.thread_ids.clear()

        return True

    def clear_all(self) -> None:
        """清空并删除全部进程内会话。"""
        with self._lock:
            states = list(
                self._sessions.values()
            )
            self._sessions.clear()

        for state in states:
            with state.lock:
                state.memory.clear()

    def session_ids(self) -> list[str]:
        """Return a stable snapshot of current session identifiers."""
        with self._lock:
            return list(self._sessions)

    def register_thread(self, session_id: str, thread_id: str) -> None:
        state = self.get_or_create(session_id)
        with state.lock:
            state.thread_ids.add(thread_id)

    def thread_ids(self, session_id: str) -> list[str]:
        clean = session_id.strip()
        with self._lock:
            state = self._sessions.get(clean)
        if state is None:
            return []
        with state.lock:
            return list(state.thread_ids)

    def all_thread_ids(self) -> list[str]:
        with self._lock:
            states = list(self._sessions.values())
        result: list[str] = []
        for state in states:
            with state.lock:
                result.extend(state.thread_ids)
        return result
