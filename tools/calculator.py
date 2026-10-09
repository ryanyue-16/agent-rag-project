from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def calculator(expression: str) -> str:
    """
    第 7 课中的简单计算器。

    适合课堂演示。生产环境应使用 AST 解析器或专门计算工具。
    """
    logger.info(
        "Calculator 工具被调用：expression=%s",
        expression,
    )

    try:
        result = eval(
            expression,
            {"__builtins__": {}},
            {},
        )
        return str(result)
    except Exception as exc:
        logger.warning(
            "Calculator 计算失败：expression=%s error=%s",
            expression,
            exc,
        )
        return f"计算错误: {exc}"