import { useTaskList } from '../hooks'
import { statusLabel } from '../api'

export function History({ onOpen }: { onOpen: (id: string) => void }) {
  const { tasks, error, refresh } = useTaskList()

  return (
    <div>
      <div className="spread" style={{ marginBottom: 16 }}>
        <h2 style={{ margin: 0 }}>历史任务</h2>
        <button className="btn" onClick={refresh}>刷新</button>
      </div>
      {error && <div className="err">{error}</div>}
      {!error && tasks.length === 0 && <div className="empty">暂无任务记录</div>}
      {tasks.map((t) => (
        <div key={t.task_id} className="task-item" onClick={() => onOpen(t.task_id)}>
          <span className={`badge ${t.status}`}>{statusLabel(t.status)}</span>
          <span className="tq">{t.query || t.task_id}</span>
          <span className="muted" style={{ flexShrink: 0 }}>{t.progress}%</span>
        </div>
      ))}
    </div>
  )
}
