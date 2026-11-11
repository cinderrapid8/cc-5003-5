import { defineConfig } from "vite";

// Vite serves the frontend during `tauri dev`; the built bundle lands in
// dist/, which tauri.conf.json points frontendDist at.
export default defineConfig({
  root: ".",
  build: {
    outDir: "dist",
    target: "es2022",
  },
  clearScreen: false,
  server: {
    port: 5199,
    strictPort: true,
  },
});
