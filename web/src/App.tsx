import { useState } from 'react'
import { CreateForm } from './components/CreateForm'
import { ProgressView } from './components/ProgressView'
import { History } from './components/History'
import { Ops } from './components/Ops'

type View =
  | { name: 'create' }
  | { name: 'task'; taskId: string }
  | { name: 'history' }
  | { name: 'ops' }

export default function App() {
  const [view, setView] = useState<View>({ name: 'create' })

  const openTask = (taskId: string) => setView({ name: 'task', taskId })

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">🔬 Deep Research</div>
        <nav className="nav">
          <button
            className={view.name === 'create' ? 'nav-btn active' : 'nav-btn'}
            onClick={() => setView({ name: 'create' })}
          >
            新建研究
          </button>
          <button
            className={view.name === 'history' ? 'nav-btn active' : 'nav-btn'}
            onClick={() => setView({ name: 'history' })}
          >
            历史任务
          </button>
          <button
            className={view.name === 'ops' ? 'nav-btn active' : 'nav-btn'}
            onClick={() => setView({ name: 'ops' })}
          >
            运维概览
          </button>
        </nav>
      </header>

      <main className="content">
        {view.name === 'create' && <CreateForm onCreated={openTask} />}
        {view.name === 'task' && (
          <ProgressView taskId={view.taskId} onBack={() => setView({ name: 'history' })} />
        )}
        {view.name === 'history' && <History onOpen={openTask} />}
        {view.name === 'ops' && <Ops />}
      </main>
    </div>
  )
}
