import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
// The site reads ../level-*/results_*.json, ../analysis/results_*.json and a few repo docs at build time,
// so every recorded number on the page comes from the repo, never typed by hand.
export default defineConfig({
  base: '/multimodal-lab/',
  plugins: [react()],
  server: { fs: { allow: ['..'] } },
});
