import { useState, type ReactNode } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useProgress } from '../hooks'
import {
  statusLabel,
  type Citation,
  type ProgressSnapshot,
  type ResearchStrategy,
  type SubAnswer,
} from '../api'

// 管线阶段：[节点 key, 完整名, 短名]
const STAGES: [string, string, string][] = [
  ['planner', '规划子问题', '规划'],
  ['retriever', '联网搜索', '搜索'],
  ['content_extractor', '抓取网页', '抓取'],
  ['source_evaluator', '评估信源', '评估'],
  ['evidence_builder', '构建证据', '证据'],
  ['analyst', '分析论证', '分析'],
  ['fact_checker', '事实核查', '核查'],
  ['report_writer', '生成报告', '报告'],
]

// 补充轮次用了 retrieving/analyzing 等别名，归一到管线 key
const ALIAS: Record<string, string> = {
  retrieving: 'retriever',
  analyzing: 'analyst',
  fact_checking: 'fact_checker',
  writing: 'report_writer',
  planning: 'planner',
  indexing: 'evidence_builder',
}

function stageIndex(status: string): number {
  if (status === 'completed') return STAGES.length
  if (status === 'pending') return -1
  const key = ALIAS[status] || status
  return STAGES.findIndex(([k]) => k === key)
}

// 步骤条下方的紧凑计数
function stageNum(key: string, d: ProgressSnapshot): string {
  switch (key) {
    case 'planner': return d.sub_questions.length ? `${d.sub_questions.length}` : ''
    case 'retriever': return d.search_queries.length ? `${d.search_queries.length}` : ''
    case 'content_extractor': return d.crawled_count ? `${d.crawled_count}` : ''
    case 'source_evaluator': return d.sources.length ? `${d.sources.length}` : ''
    case 'evidence_builder': return d.evidence_count ? `${d.evidence_count}` : ''
    case 'analyst': return d.sub_answers.length ? `${d.sub_answers.length}` : ''
    case 'fact_checker': return d.fact_check?.issues ? `${d.fact_check.issues.length}` : ''
    case 'report_writer': return d.final_report ? `${(d.final_report.length / 1000).toFixed(1)}k` : ''
    default: return ''
  }
}

// 当前 status 对应应自动展开的 section
function activeSection(status: string): string {
  if (status === 'completed') return 'report'
  const key = ALIAS[status] || status
  if (key === 'planner') return 'subq'
  if (['retriever', 'content_extractor', 'source_evaluator', 'evidence_builder'].includes(key))
    return 'sources'
  if (key === 'analyst') return 'subq'
  if (key === 'fact_checker') return 'factcheck'
  if (key === 'report_writer') return 'report'
  return 'subq'
}

// ── 轮次时间线（多轮补充研究时才显示）──
function RoundTimeline({ d }: { d: ProgressSnapshot }) {
  const rounds = d.rounds || []
  const cur = d.current_round || 1
  const terminal = d.status === 'completed' || d.status === 'failed'
  // 单轮任务无需轮次概念
  if (cur < 2 && rounds.length < 2) return null

  const maxRound = Math.max(cur, ...rounds.map((r) => r.round), 1)
  const nodes: { round: number; hist?: (typeof rounds)[number]; isCurrent: boolean }[] = []
  for (let r = 1; r <= maxRound; r++) {
    nodes.push({
      round: r,
      hist: rounds.find((x) => x.round === r),
      isCurrent: r === cur && !terminal,
    })
  }

  return (
    <div className="rounds">
      {nodes.map((n, i) => (
        <span key={n.round} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div className={`ritem ${n.isCurrent ? 'active' : n.hist ? 'done' : ''}`}>
            <span className="rdot">{n.hist && !n.isCurrent ? '✓' : n.round}</span>
            <div>
              <div className="rlabel">第 {n.round} 轮{n.isCurrent ? ' · 进行中' : ''}</div>
              {n.hist ? (
                <div className="rsummary">
                  {n.hist.fact_check_passed
                    ? '核查通过'
                    : `核查未通过 · ${n.hist.issues_count} 处问题${n.hist.follow_up_queries.length ? ' → 触发补充' : ''}`}
                </div>
              ) : n.isCurrent ? (
                <div className="rsummary">补充研究中…</div>
              ) : null}
            </div>
          </div>
          {i < nodes.length - 1 && <span className="rarrow">→</span>}
        </span>
      ))}
    </div>
  )
}

// ── 横向步骤条 ──
function Stepper({ d }: { d: ProgressSnapshot }) {
  const cur = stageIndex(d.status)
  return (
    <div className="stepper">
      {STAGES.map(([key, , short], i) => {
        const cls =
          cur === STAGES.length || i < cur ? 'done' : i === cur ? 'active' : 'pending'
        return (
          <div key={key} className={`sitem ${cls}`}>
            <span className="sdot">{cls === 'done' ? '✓' : i + 1}</span>
            <span className="slabel">{short}</span>
            <span className="smeta">{stageNum(key, d)}</span>
          </div>
        )
      })}
    </div>
  )
}

// ── 可折叠容器 ──
function Section({
  title, count, badge, open, onToggle, children,
}: {
  title: string
  count?: number | string
  badge?: ReactNode
  open: boolean
  onToggle: () => void
  children: ReactNode
}) {
  return (
    <div className="card">
      <div className="section-title clickable" onClick={onToggle}>
        <span className="caret">{open ? '▾' : '▸'}</span>
        <span>{title}</span>
        {count != null && <span className="count">{count}</span>}
        {badge && <span style={{ marginLeft: 'auto' }}>{badge}</span>}
      </div>
      {open && <div style={{ marginTop: 12 }}>{children}</div>}
    </div>
  )
}

// ── 各 section 内容（无 card 外壳，由 Section 提供）──
function StrategyBody({ s }: { s: ResearchStrategy }) {
  return (
    <>
      <div className="row" style={{ flexWrap: 'wrap', gap: 8, marginBottom: 8 }}>
        {s.intent && <span className="chip">意图：{s.intent}</span>}
        {s.domain && <span className="chip">领域：{s.domain}</span>}
        {s.depth && <span className="chip">深度：{s.depth}</span>}
        {(s.dimensions || []).map((dim) => <span key={dim} className="chip">{dim}</span>)}
      </div>
      {s.reasoning && <div className="muted" style={{ fontSize: 13 }}>{s.reasoning}</div>}
    </>
  )
}

function citeIn(id: string, reg: Citation[]) {
  return reg.find((c) => c.id === id)
}

function SubQBody({ d }: { d: ProgressSnapshot }) {
  const answerOf = new Map<string, SubAnswer>()
  d.sub_answers.forEach((a) => answerOf.set(a.sub_question_id, a))
  return (
    <>
      {d.sub_questions.map((sq) => {
        const a = answerOf.get(sq.id)
        return (
          <div key={sq.id} className="subq">
            <div className="subq-q">{sq.question}</div>
            <div className="subq-queries">
              {sq.search_queries.map((q, i) => <span key={i} className="chip">{q}</span>)}
            </div>
            {a ? (
              <>
                <div className="subq-answer">{a.answer}</div>
                <div className="subq-meta">
                  <span className="conf" style={{ color: a.confidence >= 0.7 ? 'var(--green)' : 'var(--amber)' }}>
                    置信度 {(a.confidence * 100).toFixed(0)}%
                  </span>
                  {a.evidence_gap && <span className="badge failed">证据不足</span>}
                  <span className="cites">
                    {a.citations.map((c) => {
                      const reg = citeIn(c, d.citation_registry)
                      return reg?.url ? (
                        <a key={c} className="cite" href={reg.url} target="_blank" title={reg.title}>{c}</a>
                      ) : (
                        <span key={c} className="cite" title={reg?.title || c}>{c}</span>
                      )
                    })}
                  </span>
                </div>
              </>
            ) : (
              <div className="muted" style={{ fontSize: 13 }}>
                <span className="spinner" /> 等待分析…
              </div>
            )}
          </div>
        )
      })}
    </>
  )
}

function SourcesBody({ d }: { d: ProgressSnapshot }) {
  return (
    <>
      {d.sources.map((s, i) => (
        <div key={i} className="source">
          <span className={`score ${s.accepted ? 'ok' : 'no'}`}>{s.final_score.toFixed(2)}</span>
          <span className="stitle">
            <div>{s.title || '(无标题)'}</div>
            <a className="surl" href={s.url} target="_blank">{s.url}</a>
          </span>
        </div>
      ))}
    </>
  )
}

function FactCheckBody({ d }: { d: ProgressSnapshot }) {
  const fc = d.fact_check
  const issues = fc.issues || []
  return (
    <>
      {issues.length === 0 && <div className="muted">未发现问题。</div>}
      {issues.map((it, i) => (
        <div key={i} className="issue">
          <div className="itype">{it.type}</div>
          {it.claim && <div className="iclaim">{it.claim}</div>}
          {it.reason && <div className="ireason">{it.reason}</div>}
        </div>
      ))}
      {(fc.follow_up_queries || []).length > 0 && (
        <div style={{ marginTop: 10 }}>
          <h3>补充研究查询</h3>
          <div className="subq-queries">
            {fc.follow_up_queries!.map((q, i) => <span key={i} className="chip">{q}</span>)}
          </div>
        </div>
      )}
    </>
  )
}

// ── 主视图 ──
export function ProgressView({ taskId, onBack }: { taskId: string; onBack: () => void }) {
  const { data, error } = useProgress(taskId)
  const [openMap, setOpenMap] = useState<Record<string, boolean>>({})

  if (error && !data) {
    return (
      <div>
        <button className="btn ghost" onClick={onBack}>← 返回</button>
        <div className="err" style={{ marginTop: 12 }}>{error}</div>
      </div>
    )
  }
  if (!data) return <div className="row"><span className="spinner" /> 加载中…</div>

  const active = activeSection(data.status)
  // 默认只展开当前步骤对应的 section；用户点击后固定为其选择（不再跟随进度）
  const isOpen = (k: string) => openMap[k] ?? k === active
  const toggle = (k: string) => setOpenMap((prev) => ({ ...prev, [k]: !isOpen(k) }))

  const done = data.status === 'completed'
  const failed = data.status === 'failed'
  const s = data.research_strategy
  const hasStrategy = !!(s?.intent || s?.domain)
  const fc = data.fact_check
  const hasFc = fc && fc.passed !== undefined
  const accepted = data.sources.filter((x) => x.accepted).length

  return (
    <div>
      <button className="btn ghost" onClick={onBack} style={{ marginBottom: 12 }}>← 返回</button>

      {/* 状态概览 */}
      <div className="card">
        <div className="spread">
          <div className="row">
            <span className={`badge ${data.status}`}>{statusLabel(data.status)}</span>
            <span className="muted">第 {data.current_round || 1} / {data.max_rounds} 轮</span>
          </div>
          <span className="muted">{data.progress}%</span>
        </div>
        <div className="progress-bar" style={{ marginTop: 12 }}>
          <div style={{ width: `${data.progress}%`, background: failed ? 'var(--red)' : undefined }} />
        </div>
        <div className="muted" style={{ marginTop: 8, fontSize: 13 }}>{data.progress_message}</div>
        {failed && data.error_message && <div className="err" style={{ marginTop: 10 }}>{data.error_message}</div>}
      </div>

      {/* 轮次时间线（多轮才显示）+ 当前轮步骤条 */}
      <div className="card">
        <RoundTimeline d={data} />
        {(data.current_round || 1) >= 2 && (
          <div className="round-hint">第 {data.current_round} 轮 · 当前步骤</div>
        )}
        <Stepper d={data} />
      </div>

      {/* 可折叠详情：默认展开当前步骤 */}
      {hasStrategy && (
        <Section title="研究策略" open={isOpen('strategy')} onToggle={() => toggle('strategy')}>
          <StrategyBody s={s} />
        </Section>
      )}
      {data.sub_questions.length > 0 && (
        <Section title="子问题与分析" count={data.sub_questions.length}
          open={isOpen('subq')} onToggle={() => toggle('subq')}>
          <SubQBody d={data} />
        </Section>
      )}
      {data.sources.length > 0 && (
        <Section title="信源" count={`${accepted}/${data.sources.length} 采纳`}
          open={isOpen('sources')} onToggle={() => toggle('sources')}>
          <SourcesBody d={data} />
        </Section>
      )}
      {hasFc && (
        <Section title="事实核查"
          badge={<span className={`badge ${fc.passed ? 'completed' : 'failed'}`}>
            {fc.passed ? '通过' : `未通过 · ${(fc.issues || []).length} 项`}</span>}
          open={isOpen('factcheck')} onToggle={() => toggle('factcheck')}>
          <FactCheckBody d={data} />
        </Section>
      )}
      {done && data.final_report && (
        <Section title="研究报告" open={isOpen('report')} onToggle={() => toggle('report')}>
          <div className="report">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{data.final_report}</ReactMarkdown>
          </div>
        </Section>
      )}
    </div>
  )
}
