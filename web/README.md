# Deep Research Agent — Web 前端

Deep Research Agent 的 React 前端，聚焦**调查过程透明**：实时展示研究策略、子问题分析、信源评估、事实核查与多轮补充研究的进度。

## 技术栈

React 19 + Vite 8 + TypeScript 6。刻意保持最小依赖，无路由 / 状态管理 / 图表库——用原生 `fetch` + 轮询即可。

| 用途 | 依赖 |
|------|------|
| 视图 | `react` / `react-dom` |
| 报告渲染 | `react-markdown` + `remark-gfm` |
| 数据获取 | 原生 `fetch`（`src/api.ts`）+ `setTimeout` 轮询（`src/hooks.ts`），终态自动停止 |
| 视图切换 | `App.tsx` 内状态机（create / task / history / ops），无 react-router |

## 开发

```bash
npm install
npm run dev        # http://localhost:5173
```

Vite dev server 将 `/api` 与 `/health` 代理到后端 `http://localhost:8000`（见 `vite.config.ts`），因此前端全程用相对路径，无需 CORS 配置。请确保后端 API 已启动。

```bash
npm run build      # tsc -b + vite build → dist/
npm run preview    # 预览生产构建
npm run lint       # oxlint
```

## 结构

```
web/src/
├── main.tsx                    # 入口（挂载 App，引入 styles.css）
├── App.tsx                     # 顶层视图切换
├── api.ts                      # 类型定义 + fetch 封装（对应后端 schema）
├── hooks.ts                    # useProgress（轮询）/ useTaskList
├── styles.css                  # 全站设计系统（卡片/徽章/步骤条/轮次时间线）
└── components/
    ├── CreateForm.tsx          # 创建研究任务
    ├── ProgressView.tsx        # 进度透明核心视图（步骤条 + 轮次时间线 + 可折叠详情）
    ├── History.tsx             # 任务历史列表
    └── Ops.tsx                 # 运维/健康检查
```

## 过程透明设计

`ProgressView` 把一次调查拆成可读的进度：

- **步骤条** —— 当前轮的 8 个管线节点（规划→搜索→抓取→评估→证据→分析→核查→报告），已完成打勾。
- **轮次时间线** —— 多轮补充研究时显示，`第1轮 ✓ 核查未通过·N处问题 → 触发补充` → `第2轮 ● 进行中`，避免步骤条归零造成的「进度清零」误解。
- **可折叠详情** —— 研究策略 / 子问题与分析 / 信源 / 事实核查 / 报告；默认自动展开当前步骤对应的区块，用户手动点击后固定。

数据全部来自后端 `GET /api/research/{id}/progress`。
