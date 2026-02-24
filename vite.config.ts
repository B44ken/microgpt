import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailindcss from '@tailwindcss/vite'

import { cloudflare } from "@cloudflare/vite-plugin";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailindcss(), cloudflare()],
})