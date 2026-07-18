import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// dev server 把 /api、/health 代理到 FastAPI（localhost:8000），
// 前端全程用相对路径，无需处理 CORS，也便于将来同源部署。
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
    },
  },
})
