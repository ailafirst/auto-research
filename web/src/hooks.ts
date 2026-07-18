import { useEffect, useRef, useState } from 'react'
import { api, TERMINAL, type ProgressSnapshot, type TaskStatus } from './api'

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
