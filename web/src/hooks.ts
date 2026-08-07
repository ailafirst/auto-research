import { useEffect, useRef, useState } from 'react'
import { api, TERMINAL, type Health, type ProgressSnapshot, type TaskStatus } from './api'

// 轮询任务进度快照；到终态自动停止；卸载/切换任务时清理定时器。
// 这是「原生 fetch + setTimeout」对 TanStack Query 轮询的最小替代。
export function useProgress(taskId: string | null, intervalMs = 2500) {
  const [data, setData] = useState<ProgressSnapshot | null>(null)
  const [error, setError] = useState<string | null>(null)
  const timer = useRef<number | null>(null)

  useEffect(() => {
    if (!taskId) {
      setData(null)
      return
    }
    let alive = true
    setData(null)
    setError(null)

    const tick = async () => {
      try {
        const p = await api.progress(taskId)
        if (!alive) return
        setData(p)
        setError(null)
        if (TERMINAL.has(p.status)) return // 终态：停止轮询
      } catch (e) {
        if (alive) setError((e as Error).message)
      }
      if (alive) timer.current = window.setTimeout(tick, intervalMs)
    }
    tick()

    return () => {
      alive = false
      if (timer.current) window.clearTimeout(timer.current)
    }
  }, [taskId, intervalMs])

  return { data, error }
}

// Web UI 启动即校验后端依赖（重点是模型服务连通性），之后按 intervalMs 复查。
// 容器化后 API 在容器、模型服务在宿主机，这条链路断了不会有任何报错——直到某个
// 研究任务跑到 evidence_builder 才失败。所以放在进入页面时就查，而不是等提交。
export function useHealth(intervalMs = 30000) {
  const [health, setHealth] = useState<Health | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [checking, setChecking] = useState(true)
  const timer = useRef<number | null>(null)

  useEffect(() => {
    let alive = true

    const tick = async () => {
      try {
        const h = await api.health()
        if (!alive) return
        setHealth(h)
        setError(null)
      } catch (e) {
        // /health 本身请求不通 = API 没起来，和「某个依赖坏了」是两回事
        if (alive) {
          setHealth(null)
          setError((e as Error).message)
        }
      } finally {
        if (alive) {
          setChecking(false)
          timer.current = window.setTimeout(tick, intervalMs)
        }
      }
    }
    tick()

    return () => {
      alive = false
      if (timer.current) window.clearTimeout(timer.current)
    }
  }, [intervalMs])

  return { health, error, checking }
}

// 历史任务列表；手动 refresh，另外可选轮询保持新鲜。
export function useTaskList() {
  const [tasks, setTasks] = useState<TaskStatus[]>([])
  const [error, setError] = useState<string | null>(null)

  const refresh = async () => {
    try {
      setTasks(await api.list())
      setError(null)
    } catch (e) {
      setError((e as Error).message)
    }
  }

  useEffect(() => {
    refresh()
  }, [])

  return { tasks, error, refresh }
}
