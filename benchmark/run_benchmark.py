"""基准测试运行器。

用法:
  python benchmark/run_benchmark.py                   # 运行全部 12 个任务
  python benchmark/run_benchmark.py 01               # 运行单个任务
  python benchmark/run_benchmark.py 01 03 07         # 运行指定任务
  python benchmark/run_benchmark.py --list           # 列出所有任务
  python benchmark/run_benchmark.py --api http://...  # 指定 API 地址
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

TASKS_FILE = Path(__file__).parent / "tasks.json"
RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_API = "http://localhost:8000"
POLL_INTERVAL = 5   # 秒
TIMEOUT = 600       # 单任务最长等待 10 分钟


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


async def poll_until_done(
    client: httpx.AsyncClient, api: str, task_id: str, label: str
) -> dict:
    deadline = time.time() + TIMEOUT
    last_msg = ""
    while time.time() < deadline:
        await asyncio.sleep(POLL_INTERVAL)
        resp = await client.get(f"{api}/api/research/{task_id}/status")
        s = resp.json()
        msg = f"[{s['progress']}%] {s['progress_message']}"
        if msg != last_msg:
            print(f"  {label} >> {msg}")
            last_msg = msg
        if s["status"] in ("completed", "failed"):
            return s
    raise TimeoutError(f"任务 {task_id} 超时（{TIMEOUT}s）")


async def fetch_detail(client: httpx.AsyncClient, api: str, task_id: str) -> dict:
    resp = await client.get(f"{api}/api/research/{task_id}")
    return resp.json()


async def run_single(
    client: httpx.AsyncClient,
    api: str,
    task: dict,
) -> dict:
    label = f"[{task['id']}] {task['name']}"
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  问题: {task['query']}")
    print(f"  参数: report_type={task['report_type']}, search_depth={task['search_depth']}, "
          f"max_rounds={task['max_rounds']}")
    print(f"{'=' * 60}")

    start = time.time()
    task_id = await submit_task(client, api, task)
    print(f"  任务已提交: {task_id}")

    final_status = await poll_until_done(client, api, task_id, label)
    elapsed = time.time() - start

    detail = await fetch_detail(client, api, task_id)

    # 读取报告文件（通过 task_id 推断路径）
    report_path = Path("output/tasks") / f"{task_id}.json"
    persisted: dict = {}
    if report_path.exists():
        persisted = json.loads(report_path.read_text(encoding="utf-8"))

    result = {
        "benchmark_task_id": task["id"],
        "benchmark_task_name": task["name"],
        "task_id": task_id,
        "query": task["query"],
        "params": {
            "report_type": task["report_type"],
            "search_depth": task["search_depth"],
            "max_rounds": task["max_rounds"],
            "language": task["language"],
        },
        "status": final_status["status"],
        "elapsed_seconds": round(elapsed, 1),
        "progress": final_status.get("progress", 0),
        "final_report": persisted.get("final_report", ""),
        "report_length": len(persisted.get("final_report", "")),
        "run_at": datetime.now().isoformat(),
    }

    status_str = "OK 完成" if result["status"] == "completed" else "NG 失败"
    print(f"  {status_str}  耗时 {elapsed:.0f}s  报告长度 {result['report_length']} 字符")

    return result


def save_result(result: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{result['benchmark_task_id']}_{ts}.json"
    path = RESULTS_DIR / fname
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


async def main(task_ids: list[str], api: str) -> None:
    all_tasks = load_tasks()

    if task_ids:
        selected = [t for t in all_tasks if t["id"] in task_ids]
        missing = set(task_ids) - {t["id"] for t in selected}
        if missing:
            print(f"未找到任务 ID: {missing}", file=sys.stderr)
    else:
        selected = all_tasks

    print(f"\n将运行 {len(selected)} 个任务，API: {api}")

    async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
        # 健康检查
        try:
            resp = await client.get(f"{api}/health")
            resp.raise_for_status()
        except Exception as e:
            print(f"API 不可达 ({api}): {e}", file=sys.stderr)
            sys.exit(1)

        summary: list[dict] = []
        for task in selected:
            try:
                result = await run_single(client, api, task)
                path = save_result(result)
                result["saved_to"] = str(path)
                summary.append(result)
            except Exception as e:
                print(f"  NG 任务 [{task['id']}] 执行出错: {e}", file=sys.stderr)
                summary.append({
                    "benchmark_task_id": task["id"],
                    "benchmark_task_name": task["name"],
                    "status": "error",
                    "error": str(e),
                })

    # 打印摘要
    print(f"\n{'=' * 60}")
    print("基准测试摘要")
    print(f"{'=' * 60}")
    print(f"{'ID':>4}  {'名称':<24}  {'状态':<8}  {'耗时':>8}  {'报告长度':>10}")
    print("-" * 60)
    for r in summary:
        status = "OK" if r.get("status") == "completed" else "NG"
        elapsed = f"{r.get('elapsed_seconds', 0):.0f}s" if "elapsed_seconds" in r else "-"
        length = str(r.get("report_length", "-"))
        print(f"{r['benchmark_task_id']:>4}  {r['benchmark_task_name']:<24}  {status:<8}  {elapsed:>8}  {length:>10}")

    done = sum(1 for r in summary if r.get("status") == "completed")
    print(f"\n完成: {done}/{len(summary)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="深度研究基准测试")
    parser.add_argument("tasks", nargs="*", help="任务 ID（如 01 03 07），不填则运行全部")
    parser.add_argument("--api", default=DEFAULT_API, help=f"API 地址（默认 {DEFAULT_API}）")
    parser.add_argument("--list", action="store_true", help="列出所有任务")
    args = parser.parse_args()

    if args.list:
        print_tasks(load_tasks())
        sys.exit(0)

    asyncio.run(main(args.tasks, args.api))
