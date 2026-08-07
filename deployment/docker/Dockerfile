# 多阶段构建，产出两个镜像（用 --target 选择）：
#
#   app  —— API 与 worker 共用。刻意不含 torch：配了 MODEL_SERVICE_URL 之后这两个
#           进程全程 HTTP 调用模型服务，进程内不 import torch（见 requirements.txt
#           顶部说明）。装上会让镜像多约 5GB，且让「模型服务挂了」退化成静默的本地
#           加载而不是明确报错。
#   web  —— Nginx + 构建好的前端静态资源。
#
# 模型服务不在这里：它需要 GPU 和 7.1GB 权重，留在宿主机上跑（deployment/start-all.ps1）。

# ── 前端构建 ────────────────────────────────────────────────────────────────────
FROM node:22-alpine AS web-build

WORKDIR /build
# 先只拷贝清单，让 npm ci 这一层能被 Docker 缓存住——改前端源码不会重装依赖
COPY web/package.json web/package-lock.json ./
RUN npm ci

COPY web/ ./
RUN npm run build


# ── Python 依赖 ────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS py-deps

# lxml / trafilatura 需要编译工具链；装完即随构建阶段丢弃，不进最终镜像
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --prefix=/install -r /tmp/requirements.txt


# ── 应用镜像（api / worker 共用）────────────────────────────────────────────────
FROM python:3.11-slim AS app

# curl 供 compose healthcheck 用
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=py-deps /install /usr/local

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    # fastembed 默认把 ONNX 权重下到工作目录；显式指向挂载卷，否则每次重建镜像
    # 都要重下（rag_service._get_fastembed_model 目前没传 cache_dir）
    FASTEMBED_CACHE_PATH=/models/fastembed \
    HF_HOME=/models/huggingface

COPY app/ ./app/
COPY pyproject.toml ./

# 非 root 运行；/models 与 /data 是挂载点，需可写
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /models /data \
    && chown -R appuser:appuser /app /models /data
USER appuser

EXPOSE 8000

# 默认起 API；worker 在 compose 里用 command 覆盖成 arq
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


# ── 前端镜像 ───────────────────────────────────────────────────────────────────
FROM nginx:1.27-alpine AS web

COPY --from=web-build /build/dist /usr/share/nginx/html
COPY deployment/docker/nginx.conf /etc/nginx/nginx.conf

EXPOSE 80
