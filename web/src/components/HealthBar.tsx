import { useState } from 'react'
import { BLOCKING_DEPS, DEP_LABEL } from '../api'
import { useHealth } from '../hooks'

// 顶栏右侧的依赖状态指示。三种形态：
//   正常   —— 一颗绿点，不打扰
//   降级   —— 黄条，某项依赖坏了但研究仍能跑（Redis/队列/向量库都能降级）
//   阻断   —— 红条，模型服务不可达，提交任务必然失败，明确劝阻
// 展开后列出每一项的状态。detail（内部主机名/端口/版本号）默认为空——/health 公网
// 可达，明细要带 X-Health-Token 才返回，浏览器这一路拿不到是预期行为，不是坏了。
export function HealthBar() {
  const { health, error, checking } = useHealth()
  const [open, setOpen] = useState(false)

  if (checking && !health && !error) {
    return <span className="health health-checking">检查依赖…</span>
  }

  // /health 请求本身失败 = API 不可达
  if (error || !health) {
    return (
      <span className="health health-down" title={error ?? ''}>
        ● 后端不可达
      </span>
    )
  }

  // 兼容旧版后端：/health 曾只返回 {status, version, qdrant_*}，没有这两个字段。
  // 前端可能先于后端发布（镜像分开构建），少一个 ?? [] 就是整个页面白屏。
  const failed = health.failed ?? []
  const dependencies = health.dependencies ?? []

  const blocking = failed.filter((n) => BLOCKING_DEPS.has(n))
  const degraded = failed.filter((n) => !BLOCKING_DEPS.has(n))
  const tone = blocking.length ? 'health-down' : degraded.length ? 'health-warn' : 'health-ok'

  const label = blocking.length
    ? `${blocking.map((n) => DEP_LABEL[n] ?? n).join('、')}不可用`
    : degraded.length
      ? `${degraded.map((n) => DEP_LABEL[n] ?? n).join('、')}降级`
      : '依赖正常'

  return (
    <div className="health-wrap">
      <button className={`health ${tone}`} onClick={() => setOpen(!open)}>
        ● {label}
      </button>
      {open && (
        <div className="health-panel">
          {dependencies.map((d) => (
            <div key={d.name} className="health-row">
              <span
                className={
                  d.skipped ? 'health-dot skipped' : d.ok ? 'health-dot ok' : 'health-dot bad'
                }
              />
              <span className="health-name">{DEP_LABEL[d.name] ?? d.name}</span>
              <span className="health-detail">{d.detail}</span>
            </div>
          ))}
          {blocking.length > 0 && (
            <div className="health-hint">
              模型服务不可达时，研究任务会在证据构建阶段失败。请先确认模型服务已启动
              （宿主机 <code>uvicorn app.model_server:app --port 8100</code>）。
            </div>
          )}
          {failed.length > 0 && !dependencies.some((d) => d.detail) && (
            <div className="health-hint">
              失败原因未随响应返回（<code>/health</code> 公网可达，明细默认隐藏）。
              在宿主机上看：<code>docker compose -f deployment/docker/compose.yaml logs api</code>。
            </div>
          )}
        </div>
      )}
    </div>
  )
}
