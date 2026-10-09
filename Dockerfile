FROM python:3.13-slim

# python 在容器中的常用运行设置：
# 1. 不生成__pycache__ / .pyc;
# 2. 日志直接输出到终端，不做缓冲；
# 3. 关闭 pip 版本检查，减少无关输出。
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 先复制依赖文件，再安装依赖
# requirements.txt 没有改变时，Docker 可以复用这一层的缓存
# 避免每次修改 Python 代码都重新安装全部依赖。
COPY requirements.txt ./requirements.txt

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.7,<3" \
    && python -m pip install --no-cache-dir -r requirements.txt

# 只复制运行 API 所需的项目代码
COPY api ./api
COPY app ./app
COPY agents ./agents
COPY rag ./rag
COPY tools ./tools
COPY evals ./evals
COPY main.py ./main.py

# PDF 不打包进镜像
# 运行容器时由宿主机只读挂载到这个目录
RUN mkdir -p /app/data/pdfs /app/data/indexes
EXPOSE 8000

# 使用 /ready 验证工作流与持久化知识库均已加载
# 只是用 Python 标准库，不额外安装 curl
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=3)"

# 容器内部必须监听 0.0.0.0
# 如果仍然监听127.0.0.1，宿主机无法通过端口映射访问服务
CMD ["python", "-m", "uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers"]
