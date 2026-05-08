import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: '0.0.0.0',
    proxy: {
      '/api': {
        target: 'http://localhost:3002',
        changeOrigin: true,
      },
      '/langgraph': {
        target: 'http://localhost:3002',
        changeOrigin: true,
      },
      '/workflow': {
        target: 'http://localhost:3002',
        changeOrigin: true,
      },
      '/research': {
        target: 'http://localhost:3002',
        changeOrigin: true,
      },
    },
  },
})
