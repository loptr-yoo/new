import path from 'path';
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
    const _env = loadEnv(mode, '.', 'VITE_');
    return {
      server: {
        port: 5173,
        host: 'localhost',
        proxy: {
          '/api/generate/stream': {
            target: 'http://localhost:8000',
            changeOrigin: true,
            ws: false
          },
          '/api': {
            target: 'http://localhost:8000',
            changeOrigin: true,
          }
        }
      },
      plugins: [react()],
      resolve: {
        alias: {
          '@': path.resolve(__dirname, 'src'),
        }
      },
      test: {
        environment: 'node',
        include: ['src/**/*.test.ts'],
      },
    };
});
