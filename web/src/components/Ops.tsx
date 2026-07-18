import { useTaskList } from '../hooks'
import { statusLabel, TERMINAL } from '../api'

// 运维概览：从任务列表聚合统计。缓存命中 / worker 吞吐等深入指标需后端加端点，暂略。
export function Ops() {
  const { tasks, error, refresh } = useTaskList()

  const total = tasks.length
  const completed = tasks.filter((t) => t.status === 'completed').length
  const failed = tasks.filter((t) => t.status === 'failed').length
  const running = tasks.filter((t) => !TERMINAL.has(t.status)).length
  const doneRate = total ? Math.round((completed / total) * 100) : 0

  // 按当前节点状态分布
  const dist = new Map<string, number>()
  tasks.forEach((t) => dist.set(t.status, (dist.get(t.status) || 0) + 1))

  return (
    <div>
      <div className="spread" style={{ marginBottom: 16 }}>
        <h2 style={{ margin: 0 }}>运维概览</h2>
        <button className="btn" onClick={refresh}>刷新</button>
      </div>
      {error && <div className="err">{error}</div>}

      <div className="stat-grid" style={{ marginBottom: 16 }}>
        <div className="stat"><div className="num">{total}</div><div className="lbl">总任务</div></div>
        <div className="stat"><div className="num" style={{ color: 'var(--green)' }}>{completed}</div><div className="lbl">已完成</div></div>
        <div className="stat"><div className="num" style={{ color: 'var(--primary)' }}>{running}</div><div className="lbl">进行中</div></div>
        <div className="stat"><div className="num" style={{ color: 'var(--red)' }}>{failed}</div><div className="lbl">失败</div></div>
        <div className="stat"><div className="num">{doneRate}%</div><div className="lbl">完成率</div></div>
      </div>

      <div className="card">
        <div className="section-title">状态分布</div>
        {[...dist.entries()].map(([st, n]) => (
          <div key={st} className="row" style={{ marginBottom: 8 }}>
            <span className={`badge ${st}`} style={{ minWidth: 72 }}>{statusLabel(st)}</span>
            <div className="progress-bar" style={{ flex: 1 }}>
              <div style={{ width: `${total ? (n / total) * 100 : 0}%` }} />
            </div>
            <span className="muted" style={{ minWidth: 30, textAlign: 'right' }}>{n}</span>
          </div>
        ))}
        {total === 0 && <div className="empty">暂无数据</div>}
      </div>
    </div>
  )
}
