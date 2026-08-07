"""依赖健康检查 —— 启动校验与 /health 端点共用同一份探测逻辑。

存在的理由：容器化后 API/worker 镜像不含 torch，模型服务不可达时 rag_service 里那条
「回退本地加载」的降级路径会直接 ImportError，而不是像宿主机部署那样静默变慢。
与其让第一个研究任务在 evidence_builder 处炸掉，不如在进程启动时就把结论摆出来。

设计约束：
  - **不抛异常**。每个探测各自吞掉自己的错误，任何一项失败都不能拖垮 /health，
    否则容器 healthcheck 会把一个「只是 Redis 没起来」的 API 判成 unhealthy 并重启。
  - **/health 恒返回 200**，用 body 里的 status=ok|degraded 表达健康度。存活与就绪
    是两件事：进程活着就该返回 200，依赖缺失通过 degraded 表达。
  - **超时要短**。healthcheck 每几秒跑一次，探测必须比探测间隔快得多。
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from typing import TYPE_CHECKING, Any

from app.core.config import mask_dsn, settings

if TYPE_CHECKING:
    from app.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)

# 探测超时远小于 compose healthcheck 的 interval，避免探测本身把检查拖超时
_PROBE_TIMEOUT = 3.0

# 探测专用的 VectorStoreService，跨请求复用。
#
# 每次探测都新建一个的话，构造函数会建一个 AsyncQdrantClient、_ensure_collection 会
# 再发两次 get_collections 和两次 create_payload_index，然后整个丢掉——而 /health 的
# 调用方是容器 healthcheck（15s 一次）和前端状态条（30s 一次），等于稳定地给 Qdrant
# 压一份纯属浪费的流量。内存模式下更离谱：每次探测都新建再销毁一个进程内 Qdrant 实例。
#
# 复用之后 _collection_initialized 在首次探测后即为 True，后续探测只剩 health_check()
# 和 info() 两次往返。
_probe_store: VectorStoreService | None = None
_probe_lock = asyncio.Lock()


async def _get_probe_store() -> VectorStoreService:
    """取（必要时新建）探测用的 VectorStoreService。"""
    global _probe_store

    async with _probe_lock:
        store = _probe_store
        # 缓存实例已回退内存、而配置要求 remote：丢弃重建，给远程一次重连机会。
        # 不这么做的话首次探测赶上 Qdrant 没起来，这个进程就会一直上报 memory,
        # 哪怕 Qdrant 早就恢复了——复用带来的状态粘滞必须显式解开。
        if store is not None and settings.qdrant_mode == "remote" and store.mode == "memory":
            try:
                await store.close()
            except Exception:
                pass
            store = None

        if store is None:
            from app.services.vector_store import VectorStoreService

            store = VectorStoreService()
        _probe_store = store
        return store


async def close_probe_store() -> None:
    """进程关闭时释放探测客户端持有的连接池。"""
    global _probe_store

    async with _probe_lock:
        if _probe_store is not None:
            try:
                await _probe_store.close()
            except Exception:
                pass
            _probe_store = None


async def _probe_model_service() -> dict[str, Any]:
    """模型服务连通性 —— 容器化部署下最关键的一项。

    未配置 model_service_url 时视为「本进程自带模型」，报 skipped 而非失败。
    """
    if not settings.model_service_url:
        return {
            "name": "model_service",
            "ok": True,
            "skipped": True,
            "detail": "未配置 MODEL_SERVICE_URL，模型在本进程内加载",
        }

    try:
        from app.services.rag_service import _get_http_client

        client = _get_http_client()
        resp = await asyncio.wait_for(client.get("/health"), timeout=_PROBE_TIMEOUT)
        resp.raise_for_status()
        body = resp.json()
        models = body.get("models", {})
        missing = [k for k, v in models.items() if not v]
        return {
            "name": "model_service",
            "ok": not missing,
            "detail": (
                f"{settings.model_service_url} device={body.get('device')} "
                f"dtype={body.get('dtype')}"
                + (f" 未就绪的模型: {','.join(missing)}" if missing else "")
            ),
        }
    except Exception as exc:
        return {
            "name": "model_service",
            "ok": False,
            "detail": f"{settings.model_service_url} 不可达: {type(exc).__name__}: {exc}",
        }


def _qdrant_version_mismatch(client_ver: str, server_ver: str) -> bool:
    """qdrant-client 的兼容规则：major 必须相同，minor 相差不超过 1。

    这条必须单独查，因为「连得上」并不等于「能用」：实测 client 1.19 对 server
    1.12 时 get_collections 正常返回、集合也能读到，但 upsert 直接失败（异常消息
    还是空的）。表现是研究任务照常跑完，只是 RAG 证据整段被静默跳过、报告没有引用。
    """
    try:
        c_major, c_minor = (int(x) for x in client_ver.split(".")[:2])
        s_major, s_minor = (int(x) for x in server_ver.split(".")[:2])
    except (ValueError, TypeError):
        return False   # 版本号解析不了就不下判断，避免误报
    return c_major != s_major or abs(c_minor - s_minor) > 1


async def _probe_qdrant() -> dict[str, Any]:
    # 内存模式下不做任何 I/O：实例在本进程内，建出来必然连得上，探测它没有信息量，
    # 却要付一次「新建 + 建集合 + 建索引 + 销毁」的代价。直接如实上报即可。
    if settings.qdrant_mode != "remote":
        return {
            "name": "qdrant",
            "ok": True,
            "mode": "memory",
            "detail": "进程内内存实例，向量不持久化（QDRANT_MODE=memory）",
        }

    try:
        vs = await _get_probe_store()

        # 走 _ensure_collection 而非裸 health_check：remote 配置下这里才会触发回退
        # 逻辑，mode 也才反映真实生效的模式而非配置值。复用实例时它在首次之后是空转。
        await asyncio.wait_for(vs._ensure_collection(), timeout=_PROBE_TIMEOUT)
        ok = await vs.health_check()
        mode = vs.mode

        if mode == "memory":
            return {
                "name": "qdrant",
                "ok": False,
                "mode": mode,
                "detail": f"配置 remote({settings.qdrant_url}) 但已回退内存模式",
            }

        import importlib.metadata as metadata

        client_ver = metadata.version("qdrant-client")
        info = await asyncio.wait_for(vs.client.info(), timeout=_PROBE_TIMEOUT)
        server_ver = str(getattr(info, "version", "") or "")
        if _qdrant_version_mismatch(client_ver, server_ver):
            return {
                "name": "qdrant",
                "ok": False,
                "mode": mode,
                "detail": (
                    f"客户端 {client_ver} 与服务端 {server_ver} 版本不兼容，"
                    f"写入会失败且证据构建会被静默跳过；请对齐 compose 里的 "
                    f"qdrant 镜像版本"
                ),
            }

        return {
            "name": "qdrant",
            "ok": ok,
            "mode": mode,
            "detail": f"mode=remote client={client_ver} server={server_ver}",
        }
    except Exception as exc:
        return {
            "name": "qdrant",
            "ok": False,
            "mode": "unavailable",
            "detail": f"{type(exc).__name__}: {exc}",
        }


async def _probe_redis() -> dict[str, Any]:
    if not settings.redis_url:
        return {
            "name": "redis",
            "ok": True,
            "skipped": True,
            "detail": "未配置 REDIS_URL，缓存/队列/限流均降级为进程内",
        }
    try:
        from app.services.cache_service import get_cache

        ok = await asyncio.wait_for(get_cache().ping(), timeout=_PROBE_TIMEOUT)
        return {"name": "redis", "ok": bool(ok), "detail": mask_dsn(settings.redis_url)}
    except Exception as exc:
        return {"name": "redis", "ok": False, "detail": f"{type(exc).__name__}: {exc}"}


async def _probe_database() -> dict[str, Any]:
    try:
        from sqlalchemy import text

        from app.services.db import get_sessionmaker

        async def _ping() -> None:
            async with get_sessionmaker()() as session:
                await session.execute(text("SELECT 1"))

        await asyncio.wait_for(_ping(), timeout=_PROBE_TIMEOUT)
        return {"name": "database", "ok": True, "detail": mask_dsn(settings.database_url)}
    except Exception as exc:
        return {"name": "database", "ok": False, "detail": f"{type(exc).__name__}: {exc}"}


async def _probe_queue() -> dict[str, Any]:
    if not settings.queue_enabled:
        return {
            "name": "queue",
            "ok": True,
            "skipped": True,
            "detail": "队列未启用，任务进程内执行",
        }
    try:
        from app.services.queue import get_arq_pool

        pool = await asyncio.wait_for(get_arq_pool(), timeout=_PROBE_TIMEOUT)
        return {
            "name": "queue",
            "ok": pool is not None,
            "detail": "arq 已连接" if pool is not None else "arq 不可用，回退进程内执行",
        }
    except Exception as exc:
        return {"name": "queue", "ok": False, "detail": f"{type(exc).__name__}: {exc}"}


async def collect_health() -> dict[str, Any]:
    """并发探测全部依赖，返回聚合结果。任何单项失败都不会抛出。"""
    probes = await asyncio.gather(
        _probe_model_service(),
        _probe_qdrant(),
        _probe_redis(),
        _probe_database(),
        _probe_queue(),
        return_exceptions=True,
    )

    deps: list[dict[str, Any]] = []
    for item in probes:
        if isinstance(item, BaseException):
            deps.append(
                {"name": "unknown", "ok": False, "detail": f"探测自身异常: {item}"}
            )
        else:
            deps.append(item)

    # skipped 的项目不参与健康判定：它们是「按配置就不该存在」，不是「坏了」
    failed = [d["name"] for d in deps if not d.get("ok") and not d.get("skipped")]
    qdrant = next((d for d in deps if d["name"] == "qdrant"), {})

    return {
        "status": "degraded" if failed else "ok",
        "failed": failed,
        "qdrant_mode": qdrant.get("mode", "unavailable"),
        "qdrant_connected": bool(qdrant.get("ok")),
        "dependencies": deps,
    }


def detail_allowed(token: str | None) -> bool:
    """调用方是否有权看 dependencies[].detail。

    默认（HEALTH_DETAIL_TOKEN 为空）一律不给：/health 经 Nginx 公网可达，而 detail
    里是内部主机名、端口、数据库用户名、组件版本号，失败时还有异常原文。这些对排查
    有用，但排查该走 `docker compose logs`——启动时的 verify_on_startup 已经把同样
    的内容完整写进日志，那里本来就需要宿主机权限才看得到。
    """
    expected = settings.health_detail_token
    if not expected:
        return False
    if not token:
        return False
    # 定长比较，避免按字符提前返回泄漏令牌前缀
    return secrets.compare_digest(token, expected)


def redact_health(result: dict[str, Any]) -> dict[str, Any]:
    """抹掉 detail，保留调用方判断「哪一项坏了」所需的最小信息。

    name / ok / skipped 必须留着：前端状态条据此区分「模型服务不可用（阻断）」和
    「Redis 降级（还能跑）」，这点信息本身不构成情报。
    """
    return {
        **result,
        "dependencies": [{**dep, "detail": ""} for dep in result["dependencies"]],
    }


async def verify_on_startup() -> dict[str, Any]:
    """启动时校验并把结论写进日志。

    模型服务不可达时用 error 级别：容器镜像里没有 torch，这不是「会变慢」而是
    「研究任务一定会失败」，必须和其他可降级的依赖区分开。
    """
    result = await collect_health()

    for dep in result["dependencies"]:
        name, detail = dep["name"], dep.get("detail", "")
        if dep.get("skipped"):
            logger.info("[依赖校验] %-14s 跳过 — %s", name, detail)
        elif dep.get("ok"):
            logger.info("[依赖校验] %-14s 正常 — %s", name, detail)
        elif name == "model_service":
            logger.error(
                "[依赖校验] %-14s 失败 — %s；本进程若不含 torch，研究任务会在 "
                "evidence_builder 阶段直接失败，请先确认模型服务已启动",
                name, detail,
            )
        else:
            logger.warning("[依赖校验] %-14s 失败 — %s（该项支持降级）", name, detail)

    if result["status"] == "ok":
        logger.info("[依赖校验] 全部依赖就绪")
    else:
        logger.warning("[依赖校验] 存在失败项: %s", ", ".join(result["failed"]))

    return result
