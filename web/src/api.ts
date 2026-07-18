// API 客户端 —— 原生 fetch，无第三方请求库。
// 全程用相对路径，dev 由 Vite proxy 转发到 FastAPI（见 vite.config.ts）。

// ── 类型（对应后端 app/models/task.py 的 schema）──────────────────────────────

export interface ResearchStrategy {
  intent?: string
  domain?: string
  depth?: string
  dimensions?: string[]
  reasoning?: string
}

export interface SubQuestion {
  id: string
  question: string
  search_queries: string[]
}

export interface Source {
  url: string
  title: string
  content_length?: number
  final_score: number
  accepted: boolean
  reason: string
}

export interface Citation {
  id: string
  title: string
  url: string
}

export interface SubAnswer {
  sub_question_id: string
  question: string
  answer: string
  citations: string[]
  confidence: number
  evidence_gap: boolean
}

export interface FactCheckIssue {
  type: string
  claim?: string
  reason?: string
}

export interface FactCheck {
  passed?: boolean
  issues?: FactCheckIssue[]
  follow_up_queries?: string[]
}

export interface SearchSummary {
  sq_id: string
  answer: string
}

export interface RoundSummary {
  round: number
  fact_check_passed: boolean
  issues_count: number
  follow_up_queries: string[]
}

// GET /api/research/{id}/progress —— 过程透明的完整快照
export interface ProgressSnapshot {
  task_id: string
  status: string
  progress: number
  progress_message: string
  current_round: number
  max_rounds: number
  research_strategy: ResearchStrategy
  sub_questions: SubQuestion[]
  search_queries: string[]
  search_summaries: SearchSummary[]
  sources: Source[]
  crawled_count: number
  evidence_count: number
  citation_registry: Citation[]
  sub_answers: SubAnswer[]
  fact_check: FactCheck
  fact_check_passed: boolean
  follow_up_queries: string[]
  rounds: RoundSummary[]
  final_report: string | null
  error_message: string | null
  updated_at: string
}

export interface TaskStatus {
  task_id: string
  query?: string
  status: string
  progress: number
  progress_message: string
  current_round: number
  max_rounds: number
}

// 节点状态 → 中文标签（管线阶段与徽章共用）
export const STATUS_LABEL: Record<string, string> = {
  pending: '排队中',
  planning: '规划中',
  planner: '规划子问题',
  retriever: '联网搜索',
  retrieving: '联网搜索',
  content_extractor: '抓取网页',
  source_evaluator: '评估信源',
  evidence_builder: '构建证据',
  analyst: '分析论证',
  analyzing: '分析论证',
  fact_checker: '事实核查',
  fact_checking: '事实核查',
  report_writer: '生成报告',
  writing: '生成报告',
  completed: '已完成',
  failed: '失败',
}

export function statusLabel(s: string): string {
  return STATUS_LABEL[s] || s
}

export interface CreateRequest {
  query: string
  max_rounds: number
  language: string
  report_type: string
  search_depth: string
  top_k?: number
  enable_fact_check?: boolean
}

// 终态：不再轮询
export const TERMINAL = new Set(['completed', 'failed'])

// ── fetch 封装 ────────────────────────────────────────────────────────────────

async function req<T>(method: string, path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const j = await res.json()
      if (j?.detail) detail = j.detail
    } catch {
      /* 非 JSON 错误体，用状态码 */
    }
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

export const api = {
  create: (r: CreateRequest) =>
    req<{ task_id: string; status: string }>('POST', '/api/research', r),
  progress: (id: string) =>
    req<ProgressSnapshot>('GET', `/api/research/${id}/progress`),
  list: () => req<TaskStatus[]>('GET', '/api/research'),
  health: () => req<{ status: string }>('GET', '/health'),
}
