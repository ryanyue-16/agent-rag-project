def test_runtime_dependencies_import() -> None:
    """检查项目关键依赖能否正常导入。"""
    import faiss
    import httpx
    import numpy
    import openai
    import pypdf
    import python_multipart
    import sentence_transformers
    import uvicorn
    from fastapi import FastAPI
    from fastapi.testclient import (
        TestClient,
    )
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
    )
    from langchain_openai import (
        ChatOpenAI,
    )
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import StateGraph

    assert faiss is not None
    assert httpx is not None
    assert numpy is not None
    assert openai is not None
    assert python_multipart is not None
    assert pypdf is not None
    assert sentence_transformers is not None
    assert uvicorn is not None

    assert FastAPI is not None
    assert TestClient is not None
    assert StateGraph is not None
    assert InMemorySaver is not None
    assert AIMessage is not None
    assert HumanMessage is not None
    assert ChatOpenAI is not None
