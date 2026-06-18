"""
Streamlit Web UI — Deep Research Agent 用户界面。

启动方式:
    streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

import time
import urllib.request
import urllib.error
import json as json_mod

import streamlit as st

# --- 页面配置 ---
st.set_page_config(
    page_title="Deep Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

API_BASE_URL = "http://localhost:8000"

# --- 自定义 CSS ---
st.markdown("""
<style>
    .report-header { font-size: 1.5rem; font-weight: bold; margin-bottom: 1rem; }
    .status-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 0.3rem;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .status-pending { background: #f0f0f0; color: #666; }
    .status-planning { background: #e3f2fd; color: #1565c0; }
    .status-retriever,
    .status-retrieving { background: #fff3e0; color: #e65100; }
    .status-content_extractor,
    .status-source_evaluator,
    .status-evidence_builder { background: #f3e5f5; color: #7b1fa2; }
    .status-analyst { background: #e8f5e9; color: #2e7d32; }
    .status-fact_checker { background: #fff8e1; color: #f57f17; }
    .status-report_writer { background: #e0f7fa; color: #00695c; }
    .status-completed { background: #e8f5e9; color: #1b5e20; }
    .status-failed { background: #ffebee; color: #c62828; }
    .source-item {
        padding: 0.3rem 0;
        border-bottom: 1px solid #eee;
        font-size: 0.85rem;
    }
    .progress-status {
        font-size: 1.1rem;
        font-weight: 500;
        padding: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# --- 工具函数 ---

def call_api(method: str, path: str, **kwargs) -> dict | None:
    """调用后端 API。"""
    url = f"{API_BASE_URL}{path}"
    try:
        data = None
        if method.upper() in ("POST", "PUT", "PATCH") and "json" in kwargs:
            body = json_mod.dumps(kwargs["json"]).encode("utf-8")
            req = urllib.request.Request(url, data=body,
                                         headers={"Content-Type": "application/json"},
                                         method=method.upper())
        else:
            req = urllib.request.Request(url, method=method.upper())

        with urllib.request.urlopen(req, timeout=30) as resp:
            return json_mod.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        st.error(f"API 错误: {exc.code} {exc.reason}")
        return None
    except Exception as exc:
        st.error(f"请求出错: {exc}")
        return None


def get_status_color(status: str) -> str:
    """获取状态对应的 CSS 类名。"""
    colors = {
        "pending": "status-pending",
        "planning": "status-planning",
        "planner": "status-planning",
        "retriever": "status-retriever",
        "retrieving": "status-retriever",
        "content_extractor": "status-content_extractor",
        "source_evaluator": "status-source_evaluator",
        "evidence_builder": "status-evidence_builder",
        "analyst": "status-analyst",
        "analyzing": "status-analyst",
        "fact_checker": "status-fact_checker",
        "fact_checking": "status-fact_checker",
        "report_writer": "status-report_writer",
        "writing": "status-report_writer",
        "completed": "status-completed",
        "failed": "status-failed",
    }
    return colors.get(status, "status-pending")


def status_to_cn(status: str) -> str:
    """状态英文转中文。"""
    mapping = {
        "pending": "排队中",
        "planning": "规划中",
        "planner": "分析问题",
        "retriever": "搜索资料",
        "retrieving": "搜索资料",
        "content_extractor": "抓取网页",
        "source_evaluator": "评估信源",
        "evidence_builder": "构建证据",
        "indexing": "索引中",
        "analyst": "分析内容",
        "analyzing": "分析中",
        "fact_checker": "事实核查",
        "fact_checking": "核查中",
        "report_writer": "生成报告",
        "writing": "写作中",
        "completed": "已完成",
        "failed": "失败",
    }
    return mapping.get(status, status)


def render_report(report_text: str) -> None:
    """渲染 Markdown 报告。"""
    if not report_text:
        st.info("报告为空")
        return

    st.markdown(report_text)
    if st.button("📋 复制报告"):
        st.code(report_text, language="markdown")
        st.success("请手动复制上方内容")


# --- 主页面 ---

st.title("🔬 Deep Research Agent")
st.markdown("输入一个研究问题，系统将自动搜索、分析并生成结构化研究报告。")

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 配置")

    api_url = st.text_input("API 地址", value=API_BASE_URL)

    max_rounds = st.slider("最大研究轮数", min_value=1, max_value=5, value=2,
                          help="研究轮数越多，报告越深入")
    language = st.selectbox("报告语言", options=["zh-CN", "en"], index=0)
    report_type = st.selectbox("报告类型",
                               options=["deep", "summary", "comparison"],
                               index=0,
                               format_func=lambda x: {
                                   "deep": "深度研究",
                                   "summary": "快速摘要",
                                   "comparison": "对比分析",
                               }[x])
    enable_fact_check = st.checkbox("启用事实核查", value=True)

    st.divider()

    # 历史任务
    st.header("📋 历史任务")
    if st.button("刷新任务列表"):
        tasks = call_api("GET", "/api/research")
        if tasks:
            st.session_state.tasks = tasks
        else:
            st.info("暂无任务记录")

    if "tasks" in st.session_state:
        for t in st.session_state.tasks[-10:]:
            col1, col2 = st.columns([3, 1])
            with col1:
                short_query = t.get("query", "")[:30]
                if st.button(f"{short_query}...", key=f"task_{t['task_id']}"):
                    st.session_state.selected_task = t["task_id"]
                    st.rerun()
            with col2:
                st.markdown(
                    f'<span class="status-badge {get_status_color(t["status"])}">'
                    f'{status_to_cn(t["status"])}</span>',
                    unsafe_allow_html=True,
                )

# 主内容区
tab1, tab2 = st.tabs(["🔍 新研究", "📄 查看报告"])

with tab1:
    st.header("开始新的研究")

    with st.form("research_form"):
        query = st.text_area(
            "研究问题",
            placeholder="例如：2026 年 AI Agent 行业的发展趋势有哪些？",
            height=100,
            help="请输入你想研究的主题或问题",
        )

        col1, col2 = st.columns([1, 4])
        with col1:
            submitted = st.form_submit_button("🚀 开始研究", type="primary",
                                             use_container_width=True)
        with col2:
            st.caption("系统将自动搜索、分析并生成结构化研究报告")

    if submitted:
        if not query or len(query.strip()) < 2:
            st.error("请输入有效的研究问题（至少 2 个字符）")
        else:
            result = call_api("POST", "/api/research", json={
                "query": query.strip(),
                "max_rounds": max_rounds,
                "language": language,
                "report_type": report_type,
                "enable_fact_check": enable_fact_check,
            })

            if result:
                task_id = result.get("task_id", "")
                st.success(f"✅ 任务已创建！任务 ID: {task_id}")
                st.session_state.current_task = task_id
                st.session_state.selected_task = task_id
                time.sleep(0.5)
                st.rerun()

    # 当前任务进度
    current_task_id = st.session_state.get("current_task")
    if current_task_id:
        st.divider()
        st.subheader(f"📊 任务进度")

        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        report_placeholder = st.empty()

        # 轮询任务状态（最多等 5 分钟）
        for _ in range(300):
            status_data = call_api("GET", f"/api/research/{current_task_id}/status")
            if not status_data:
                break

            task_status = status_data.get("status", "")
            progress_val = status_data.get("progress", 0)
            message = status_data.get("progress_message", "")
            current_round = status_data.get("current_round", 0)
            max_rounds = status_data.get("max_rounds", 2)

            # 进度条和状态
            with progress_placeholder.container():
                st.progress(progress_val / 100, text=message or status_to_cn(task_status))
                st.markdown(
                    f'<div class="progress-status">'
                    f'<span class="status-badge {get_status_color(task_status)}">'
                    f'{status_to_cn(task_status)}</span>'
                    f' 第 {current_round}/{max_rounds} 轮'
                    f' | 进度 {progress_val}%'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            if task_status == "completed":
                status_placeholder.success("🎉 研究完成！")
                report_data = call_api("GET", f"/api/research/{current_task_id}/report")
                if report_data and report_data.get("report"):
                    with report_placeholder.container():
                        st.divider()
                        st.subheader("📄 研究报告")
                        render_report(report_data["report"])
                break
            elif task_status == "failed":
                detail = call_api("GET", f"/api/research/{current_task_id}")
                err = detail.get("error_message", "未知错误") if detail else "未知错误"
                status_placeholder.error(f"❌ 研究失败: {err}")
                break
            elif task_status == "pending":
                status_placeholder.info("⏳ 任务排队中...")

            time.sleep(1)
        else:
            status_placeholder.warning(
                "⏰ 研究仍在进行中，请稍后刷新页面查看。"
                f"\n\n任务 ID: `{current_task_id}` "
                "可到「查看报告」页签输入此 ID 获取结果。"
            )

with tab2:
    st.header("查看研究报告")

    view_mode = st.radio("选择方式", ["从历史任务", "输入任务 ID"], horizontal=True)

    task_id_to_view = None

    if view_mode == "从历史任务":
        if "tasks" in st.session_state and st.session_state.tasks:
            task_options = {
                t["task_id"]: f"{t['task_id'][:12]}... - {t.get('query', '')[:40]}"
                for t in st.session_state.tasks
            }
            selected = st.selectbox(
                "选择任务",
                options=list(task_options.keys()),
                format_func=lambda x: task_options.get(x, x),
            )
            task_id_to_view = selected
        else:
            st.info("暂无历史任务，请先创建一个研究任务。或先在「新研究」页面刷新历史任务列表。")
    else:
        task_id_to_view = st.text_input("输入任务 ID")

    if task_id_to_view:
        detail = call_api("GET", f"/api/research/{task_id_to_view}")

        if detail:
            status = detail.get("status", "")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("状态", status_to_cn(status))
            with col2:
                st.metric("进度", f'{detail.get("progress", 0)}%')
            with col3:
                st.metric("轮数",
                          f'{detail.get("current_round", 0)}/{detail.get("max_rounds", 2)}')

            if status == "completed":
                report_data = call_api("GET", f"/api/research/{task_id_to_view}/report")
                if report_data:
                    st.divider()
                    render_report(report_data.get("report", ""))
            elif status == "failed":
                st.error(f"❌ 任务失败: {detail.get('error_message', '未知错误')}")
            else:
                st.info(f"⏳ {detail.get('progress_message', '任务执行中...')}")

            with st.expander("查看任务详情"):
                st.json(detail)
        else:
            st.error("未找到该任务")


# --- 底部 ---
st.divider()
st.caption("Deep Research Agent v0.1.0 | 基于 LangGraph + RAG + FastAPI")
