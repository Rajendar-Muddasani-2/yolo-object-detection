import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/detect': 'http://localhost:8080',
      '/health': 'http://localhost:8080',
      '/classes': 'http://localhost:8080',
      '/metrics': 'http://localhost:8080',
    },
  },
});
