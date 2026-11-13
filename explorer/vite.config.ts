import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  // development only: lets ?devroot=<absolute path to a runs folder> read records through the dev server
  // (two levels up, so sibling folders of this repo's parent, like the example runs, are readable too)
  server: { fs: { allow: ["../.."] } },
});
