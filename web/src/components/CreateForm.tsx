import { useState } from 'react'
import { api } from '../api'

export function CreateForm({ onCreated }: { onCreated: (id: string) => void }) {
  const [query, setQuery] = useState('')
  const [maxRounds, setMaxRounds] = useState(2)
  const [reportType, setReportType] = useState('deep')
  const [searchDepth, setSearchDepth] = useState('advanced')
  const [language, setLanguage] = useState('zh-CN')
  const [factCheck, setFactCheck] = useState(true)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const submit = async () => {
    if (query.trim().length < 2) {
      setError('请输入研究问题（至少 2 个字）')
      return
    }
    setSubmitting(true)
    setError(null)
    try {
      const r = await api.create({
        query: query.trim(),
        max_rounds: maxRounds,
        language,
        report_type: reportType,
        search_depth: searchDepth,
        enable_fact_check: factCheck,
      })
      onCreated(r.task_id)
    } catch (e) {
      setError((e as Error).message)
      setSubmitting(false)
    }
  }

  return (
    <div className="card" style={{ maxWidth: 720, margin: '0 auto' }}>
      <h2>新建研究任务</h2>
      <div className="form-row">
        <label>研究问题</label>
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="例如：固态电池相比液态锂电池有哪些核心优势与挑战，当前商业化进展如何"
        />
      </div>
      <div className="form-grid">
        <div className="form-row">
          <label>报告类型</label>
          <select value={reportType} onChange={(e) => setReportType(e.target.value)}>
            <option value="deep">深度研究</option>
            <option value="summary">快速摘要</option>
            <option value="comparison">对比分析</option>
          </select>
        </div>
        <div className="form-row">
          <label>搜索深度</label>
          <select value={searchDepth} onChange={(e) => setSearchDepth(e.target.value)}>
            <option value="advanced">深度（advanced）</option>
            <option value="basic">快速（basic）</option>
          </select>
        </div>
        <div className="form-row">
          <label>最大研究轮数：{maxRounds}</label>
          <input
            type="range"
            min={1}
            max={5}
            value={maxRounds}
            onChange={(e) => setMaxRounds(+e.target.value)}
          />
        </div>
        <div className="form-row">
          <label>输出语言</label>
          <select value={language} onChange={(e) => setLanguage(e.target.value)}>
            <option value="zh-CN">中文</option>
            <option value="en">English</option>
          </select>
        </div>
      </div>
      <div className="form-row">
        <label style={{ display: 'inline-flex', alignItems: 'center', gap: 8, fontWeight: 400 }}>
          <input
            type="checkbox"
            checked={factCheck}
            onChange={(e) => setFactCheck(e.target.checked)}
            style={{ width: 'auto' }}
          />
          启用事实核查（核查未通过时触发补充研究轮次）
        </label>
      </div>
      {error && <div className="err" style={{ marginBottom: 12 }}>{error}</div>}
      <button className="btn primary" onClick={submit} disabled={submitting}>
        {submitting ? '提交中…' : '开始研究'}
      </button>
    </div>
  )
}
