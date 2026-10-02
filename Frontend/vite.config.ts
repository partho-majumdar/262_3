import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev proxy keeps the browser same-origin, so the frontend never needs CORS
// and no API key or login is involved anywhere.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ''),
      },
    },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: './src/test/setup.ts',
  },
})
