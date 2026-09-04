import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// 开发服务器配置：/api 请求代理到 FastAPI 后端（免 CORS 配置）
export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
