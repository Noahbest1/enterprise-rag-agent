import {defineConfig} from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    // 5173 / 5174 are used by other projects on this machine; pin to 5714
    // to match README + docker-compose.yml's COPILOT_CORS_ALLOW_ORIGINS default.
    port: 5714,
    strictPort: true,
  },
});
