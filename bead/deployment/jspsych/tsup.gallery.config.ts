import { defineConfig } from "tsup";

export default defineConfig({
  entry: {
    "gallery-bundle": "src/gallery/gallery-bundle.ts",
  },
  format: ["iife"],
  globalName: "BeadGallery",
  dts: false,
  sourcemap: false,
  clean: false,
  target: "es2020",
  splitting: false,
  treeshake: true,
  minify: false,
  // jspsych is loaded from CDN as a global; keep it external
  external: ["jspsych"],
  outDir: "dist",
  esbuildOptions(options) {
    options.banner = {
      js: "/* @bead/jspsych-gallery - Interactive demo bundle */",
    };
  },
});
