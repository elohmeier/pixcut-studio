import { defineConfig } from "vite";
import { readFileSync } from "node:fs";
export default defineConfig({
  base: "./",
  build: { chunkSizeWarningLimit: 700 },
  plugins: [
    {
      name: "opencv-license",
      generateBundle() {
        this.emitFile({
          type: "asset",
          fileName: "licenses/opencv-js.txt",
          source:
            "OpenCV.js / @techstark/opencv-js 4.12.0-release.1\nhttps://github.com/opencv/opencv\nhttps://github.com/TechStark/opencv-js\n\n" +
            readFileSync(
              new URL(
                "./node_modules/@techstark/opencv-js/LICENSE",
                import.meta.url,
              ),
              "utf8",
            ),
        });
      },
    },
  ],
});
