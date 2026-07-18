"""多worker 基准测试运行器（arq 队列 + worker 池版）。

与旧版的区别：旧版串行(提交→等完成→下一个)，队列里永远只有 1 个作业，
测不到多worker。本版把任务按「到达模型」投递，由 worker 池并发消费，
测吞吐 / 排队 / 并行度，同时保留原有质量指标（报告长度、核查 issue）。

前置：API 已起 + 另起 worker 池
    uvicorn app.main:app --port 8000
    arq app.worker.WorkerSettings        # 多开几个 = 更大 worker 池

用法：
  python benchmark/run_benchmark.py --mode serial   --flush-cache   # 串行冷基线
  python benchmark/run_benchmark.py --mode burst     --flush-cache   # 齐发（冷）
  python benchmark/run_benchmark.py --mode arrival --arrival-interval 8   # 泊松到达（真实场景）
  python benchmark/run_benchmark.py --mode all --flush-cache              # 三种依次跑 + 对比表
  python benchmark/run_benchmark.py 01 03 --mode burst                    # 只跑指定任务
  python benchmark/run_benchmark.py --list

三种到达模型：
  serial   —— 一次一个，前一个完成才提交下一个（并行度≈1，质量回归基线）
  burst    —— t=0 齐发全部，worker 池上限决定真实并行
  arrival  —— 泊松到达（指数间隔），模拟真实用户陆续提交，暴露排队等待
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

TASKS_FILE = Path(__file__).parent / "tasks.json"
RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_API = "http://localhost:8000"
POLL_INTERVAL = 3       # 秒（比旧版 5s 密，排队/延迟分辨率更高）
TIMEOUT = 1200          # 单任务最长等待 20 分钟（含排队）
_TERMINAL = ("completed", "failed")


# ── 数据加载 ─────────────────────────────────────────────────────────────────
def load_tasks() -> list[dict]:
    return json.loads(TASKS_FILE.read_text(encoding="utf-8"))


def print_tasks(tasks: list[dict]) -> None:
    print(f"\n{'ID':>4}  {'名称':<24}  {'类型':<16}  {'报告':<12}  {'深度':<10}  说明")
    print("-" * 100)
    for t in tasks:
        print(
            f"{t['id']:>4}  {t['name']:<24}  {t['question_type']:<16}  "
            f"{t['report_type']:<12}  {t['search_depth']:<10}  {t['domain']}"
        )


# ── 缓存清理（公平冷启动）────────────────────────────────────────────────────
async def flush_cache() -> str:
    """删除搜索/embedding 缓存键，保证本次运行冷启动。

    仅清 dr:{version}:search:* 与 :emb:*，不动限流桶 / 任务进度。
    Redis 不可用则跳过（返回提示）。
    """
    try:
        from redis.asyncio import from_url

        from app.core.config import settings
    except Exception as exc:  # pragma: no cover
        return f"跳过（无法导入 redis/settings: {exc}）"

    if not settings.redis_url:
        return "跳过（REDIS_URL 未配置，缓存本就禁用）"

    try:
        client = from_url(settings.redis_url)
        deleted = 0
        for kind in ("search", "emb"):
            pattern = f"dr:{settings.cache_version}:{kind}:*"
            keys = [k async for k in client.scan_iter(match=pattern, count=500)]
            if keys:
                deleted += await client.delete(*keys)
        await client.aclose()
        return f"已清 {deleted} 个缓存键（search+emb）→ 冷启动"
    except Exception as exc:
        return f"清理失败（不阻断）: {exc}"


async def fetch_worker_pids(records: list[dict]) -> None:
    """从 Redis 旁路键读回每个任务由哪个 worker 进程执行（worker.py 写入），
    就地写入 record['worker_pid']，用于观测跨进程分摊。"""
    try:
        from redis.asyncio import from_url

        from app.core.config import settings
    except Exception:
        return
    if not settings.redis_url:
        return
    try:
        client = from_url(settings.redis_url)
        for r in records:
            tid = r.get("task_id")
            if not tid:
                continue
            v = await client.get(f"dr:jobworker:{tid}")
            if v is not None:
                r["worker_pid"] = v.decode() if isinstance(v, (bytes, bytearray)) else str(v)
        await client.aclose()
    except Exception:
        pass


# ── 单任务：投递 + 轮询（记录时间戳）─────────────────────────────────────────
async def submit_task(client: httpx.AsyncClient, api: str, task: dict) -> str:
    payload = {
        "query": task["query"],
        "max_rounds": task["max_rounds"],
        "language": task["language"],
        "report_type": task["report_type"],
        "search_depth": task["search_depth"],
    }
    resp = await client.post(f"{api}/api/research", json=payload)
    resp.raise_for_status()
    return resp.json()["task_id"]


async def run_one(
    client: httpx.AsyncClient,
    api: str,
    task: dict,
    arrival_offset: float,
    t_zero: float,
    verbose: bool = True,
) -> dict:
    """按到达偏移投递一个任务并轮询到终态，记录排队/服务/端到端时间。"""
    label = f"[{task['id']}] {task['name']}"

    # 到达调度：相对 t_zero 的偏移
    now = time.time()
    wait = (t_zero + arrival_offset) - now
    if wait > 0:
        await asyncio.sleep(wait)

    t_submit = time.time()
    try:
        task_id = await submit_task(client, api, task)
    except Exception as exc:
        return {
            "benchmark_task_id": task["id"], "benchmark_task_name": task["name"],
            "status": "error", "error": f"submit: {exc}",
        }
    if verbose:
        print(f"  ⇢ 投递 {label}  (+{arrival_offset:.1f}s)  {task_id}")

    t_start: float | None = None      # 首次离开 pending（worker 取走开始处理）
    deadline = time.time() + TIMEOUT
    final = None
    while time.time() < deadline:
        await asyncio.sleep(POLL_INTERVAL)
        try:
            s = (await client.get(f"{api}/api/research/{task_id}/status")).json()
        except Exception:
            continue
        st = s.get("status", "pending")
        if t_start is None and st not in ("pending", "queued"):
            t_start = time.time()
        if st in _TERMINAL:
            final = s
            break
    t_done = time.time()
    if final is None:
        return {
            "benchmark_task_id": task["id"], "benchmark_task_name": task["name"],
            "task_id": task_id, "status": "timeout",
            "queue_wait": None, "service_time": None, "latency": round(t_done - t_submit, 1),
        }

    # 完整产出从 API detail 读（不再依赖已废弃的 output/tasks/*.json）
    detail: dict = {}
    try:
        detail = (await client.get(f"{api}/api/research/{task_id}")).json()
    except Exception:
        pass
    report = detail.get("final_report") or ""
    fc = detail.get("fact_check_result") or {}
    fc_issues = fc.get("issues", []) if isinstance(fc, dict) else []
    fc_passed = fc.get("passed", True) if isinstance(fc, dict) else True

    queue_wait = round((t_start - t_submit), 1) if t_start else 0.0
    service_time = round((t_done - t_start), 1) if t_start else None
    latency = round(t_done - t_submit, 1)

    rec = {
        "benchmark_task_id": task["id"],
        "benchmark_task_name": task["name"],
        "task_id": task_id,
        "query": task["query"],
        "status": final.get("status"),
        "arrival_offset": round(arrival_offset, 1),
        "t_submit": round(t_submit - t_zero, 2),
        "t_start": round((t_start - t_zero), 2) if t_start else None,
        "t_done": round(t_done - t_zero, 2),
        "queue_wait": queue_wait,          # 提交→开始处理（worker 都忙时的排队）
        "service_time": service_time,      # 开始→完成（真实处理耗时）
        "latency": latency,                # 端到端（排队+服务）
        "report_length": len(report),
        "fact_check_passed": fc_passed,
        "fact_check_issues": fc_issues,
        "run_at": datetime.now().isoformat(),
    }
    if verbose:
        ok = "OK" if rec["status"] == "completed" else "NG"
        print(f"  ✓ {ok} {label}  排队 {queue_wait:.0f}s · 服务 "
              f"{service_time or 0:.0f}s · 端到端 {latency:.0f}s · 报告 {rec['report_length']}字")
    return rec


# ── 到达模型编排 ─────────────────────────────────────────────────────────────
def arrival_offsets(mode: str, n: int, interval: float, seed: int) -> list[float]:
    """生成各任务相对 t_zero 的投递偏移（秒）。"""
    if mode == "burst":
        return [0.0] * n
    if mode == "arrival":
        rng = random.Random(seed)
        offs, acc = [], 0.0
        for _ in range(n):
            offs.append(acc)
            acc += rng.expovariate(1.0 / interval)   # 指数间隔 → 泊松到达
        return offs
    return [0.0] * n   # serial 不用偏移（顺序 await）


async def run_mode(
    client: httpx.AsyncClient, api: str, tasks: list[dict],
    mode: str, interval: float, seed: int,
) -> list[dict]:
    n = len(tasks)
    t_zero = time.time()

    if mode == "serial":
        # 一次一个：前一个完成才提交下一个
        results = []
        for t in tasks:
            results.append(await run_one(client, api, t, 0.0, time.time(), verbose=True))
        return results

    offs = arrival_offsets(mode, n, interval, seed)
    if mode == "arrival":
        print(f"  泊松到达：均值间隔 {interval}s，投递时刻(s) = "
              f"{[round(o, 1) for o in offs]}")
    coros = [run_one(client, api, t, off, t_zero, verbose=True)
             for t, off in zip(tasks, offs)]
    return await asyncio.gather(*coros)


# ── 指标计算 ─────────────────────────────────────────────────────────────────
def _pct(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = max(0, min(len(s) - 1, int(round((p / 100) * (len(s) - 1)))))
    return s[k]


def peak_parallelism(records: list[dict]) -> int:
    """由各任务 [t_start, t_done] 区间重叠算真实峰值并行度。"""
    events = []
    for r in records:
        if r.get("t_start") is None or r.get("t_done") is None:
            continue
        events.append((r["t_start"], 1))
        events.append((r["t_done"], -1))
    events.sort()
    cur = peak = 0
    for _, delta in events:
        cur += delta
        peak = max(peak, cur)
    return peak


def compute_metrics(records: list[dict], mode: str) -> dict:
    done = [r for r in records if r.get("status") == "completed"]
    submits = [r["t_submit"] for r in records if r.get("t_submit") is not None]
    dones = [r["t_done"] for r in records if r.get("t_done") is not None]
    makespan = round(max(dones) - min(submits), 1) if submits and dones else 0.0
    lat = [r["latency"] for r in done if r.get("latency") is not None]
    qw = [r["queue_wait"] for r in done if r.get("queue_wait") is not None]
    svc = [r["service_time"] for r in done if r.get("service_time") is not None]
    sum_service = round(sum(svc), 1)   # 串行估计（服务时间之和）
    pids = sorted({r["worker_pid"] for r in records if r.get("worker_pid")})
    return {
        "mode": mode,
        "n_tasks": len(records),
        "n_completed": len(done),
        "makespan": makespan,
        "throughput_per_min": round(len(done) / makespan * 60, 2) if makespan else 0.0,
        "peak_parallelism": peak_parallelism(records),
        "distinct_workers": len(pids),
        "worker_pids": pids,
        "sum_service_time": sum_service,
        "speedup_vs_serial": round(sum_service / makespan, 2) if makespan else 0.0,
        "latency_mean": round(sum(lat) / len(lat), 1) if lat else 0.0,
        "latency_p50": round(_pct(lat, 50), 1),
        "latency_p95": round(_pct(lat, 95), 1),
        "queue_wait_mean": round(sum(qw) / len(qw), 1) if qw else 0.0,
        "queue_wait_max": round(max(qw), 1) if qw else 0.0,
        "service_mean": round(sum(svc) / len(svc), 1) if svc else 0.0,
    }


# ── 报告输出 ─────────────────────────────────────────────────────────────────
def print_report(records: list[dict], metrics: dict, cold: bool) -> None:
    m = metrics
    print(f"\n{'=' * 78}")
    print(f"模式 {m['mode'].upper()}  ·  缓存 {'COLD(已清)' if cold else 'WARM'}")
    print(f"{'=' * 78}")
    print(f"{'ID':>4}  {'名称':<22}  {'状态':<5}  {'排队':>6}  {'服务':>6}  "
          f"{'端到端':>7}  {'报告':>6}  {'核查':<5}  {'issues':>6}")
    print("-" * 78)
    for r in sorted(records, key=lambda x: x.get("t_submit") or 0):
        status = "OK" if r.get("status") == "completed" else "NG"
        qw = f"{r['queue_wait']:.0f}s" if r.get("queue_wait") is not None else "-"
        sv = f"{r['service_time']:.0f}s" if r.get("service_time") is not None else "-"
        la = f"{r['latency']:.0f}s" if r.get("latency") is not None else "-"
        length = str(r.get("report_length", "-"))
        fc = "通过" if r.get("fact_check_passed", True) else "未通过"
        ni = len(r.get("fact_check_issues", []))
        print(f"{r['benchmark_task_id']:>4}  {r['benchmark_task_name']:<22}  {status:<5}  "
              f"{qw:>6}  {sv:>6}  {la:>7}  {length:>6}  {fc:<5}  {ni:>6}")

    print("-" * 78)
    print(f"完成 {m['n_completed']}/{m['n_tasks']}  ·  "
          f"makespan {m['makespan']:.0f}s  ·  吞吐 {m['throughput_per_min']:.1f} 任务/分")
    print(f"峰值并行度 {m['peak_parallelism']}  ·  服务时间之和(串行估计) "
          f"{m['sum_service_time']:.0f}s  ·  加速比 {m['speedup_vs_serial']:.2f}×")
    print(f"参与 worker 进程数 {m['distinct_workers']}  ·  PIDs {m['worker_pids']}"
          f"  → {'跨进程分摊 ✓' if m['distinct_workers'] >= 2 else '单进程独吞（作业数≤单worker max_jobs）'}")
    print(f"端到端延迟  mean {m['latency_mean']:.0f}s / p50 {m['latency_p50']:.0f}s / "
          f"p95 {m['latency_p95']:.0f}s")
    print(f"排队等待    mean {m['queue_wait_mean']:.0f}s / max {m['queue_wait_max']:.0f}s  ·  "
          f"服务耗时 mean {m['service_mean']:.0f}s")


def print_comparison(all_metrics: list[dict]) -> None:
    if len(all_metrics) < 2:
        return
    print(f"\n{'=' * 78}")
    print("模式对比")
    print(f"{'=' * 78}")
    print(f"{'模式':<10}  {'完成':>6}  {'makespan':>9}  {'吞吐/分':>8}  "
          f"{'峰值并行':>8}  {'worker数':>8}  {'加速比':>7}  {'排队均值':>8}  {'延迟p95':>8}")
    print("-" * 88)
    for m in all_metrics:
        print(f"{m['mode']:<10}  {m['n_completed']:>2}/{m['n_tasks']:<3}  "
              f"{m['makespan']:>8.0f}s  {m['throughput_per_min']:>8.1f}  "
              f"{m['peak_parallelism']:>8}  {m['distinct_workers']:>8}  "
              f"{m['speedup_vs_serial']:>6.2f}×  "
              f"{m['queue_wait_mean']:>7.0f}s  {m['latency_p95']:>7.0f}s")


def save_run(records: list[dict], metrics: dict, cold: bool) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"mw_{metrics['mode']}_{ts}.json"
    path.write_text(json.dumps(
        {"metrics": metrics, "cold_cache": cold, "records": records},
        ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ── 主流程 ───────────────────────────────────────────────────────────────────
async def main(args: argparse.Namespace) -> None:
    all_tasks = load_tasks()
    if args.max_rounds is not None:
        for t in all_tasks:
            t["max_rounds"] = args.max_rounds
        print(f"  已覆盖所有任务 max_rounds = {args.max_rounds}")
    if args.tasks:
        selected = [t for t in all_tasks if t["id"] in args.tasks]
        missing = set(args.tasks) - {t["id"] for t in selected}
        if missing:
            print(f"未找到任务 ID: {missing}", file=sys.stderr)
    else:
        selected = all_tasks

    modes = ["serial", "burst", "arrival"] if args.mode == "all" else [args.mode]
    print(f"\n将运行 {len(selected)} 个任务 · 模式 {modes} · API {args.api}")

    async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
        try:
            (await client.get(f"{args.api}/health")).raise_for_status()
        except Exception as e:
            print(f"API 不可达 ({args.api}): {e}", file=sys.stderr)
            sys.exit(1)

        all_metrics: list[dict] = []
        for mode in modes:
            cold = args.flush_cache
            if cold:
                print(f"\n[{mode}] 清缓存: {await flush_cache()}")
            print(f"\n▶ 开始模式 {mode.upper()} ...")
            records = await run_mode(client, args.api, selected,
                                     mode, args.arrival_interval, args.seed)
            await fetch_worker_pids(records)          # 读回每任务的执行 worker PID
            metrics = compute_metrics(records, mode)
            path = save_run(records, metrics, cold)
            print_report(records, metrics, cold)
            print(f"  已保存: {path}")
            all_metrics.append(metrics)

        print_comparison(all_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多worker 深度研究基准测试")
    parser.add_argument("tasks", nargs="*", help="任务 ID（如 01 03），不填则全部")
    parser.add_argument("--api", default=DEFAULT_API, help=f"API 地址（默认 {DEFAULT_API}）")
    parser.add_argument("--mode", default="arrival",
                        choices=["serial", "burst", "arrival", "all"],
                        help="到达模型：serial 串行 / burst 齐发 / arrival 泊松到达 / all 三种依次")
    parser.add_argument("--arrival-interval", type=float, default=10.0,
                        help="arrival 模式泊松均值间隔（秒，默认 10）")
    parser.add_argument("--seed", type=int, default=42, help="arrival 随机种子（可复现）")
    parser.add_argument("--max-rounds", type=int, default=None,
                        help="覆盖所有任务的 max_rounds（不改 tasks.json）。设 2 可让任务"
                             "在事实核查未通过时触发补充研究轮次")
    parser.add_argument("--flush-cache", action="store_true",
                        help="每种模式前清搜索/embedding 缓存，保证公平冷启动")
    parser.add_argument("--list", action="store_true", help="列出所有任务")
    args = parser.parse_args()

    if args.list:
        print_tasks(load_tasks())
        sys.exit(0)

    asyncio.run(main(args))
