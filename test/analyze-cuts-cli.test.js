import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { PNG } from "pngjs";
import { analysisRaster } from "../src/analysis-raster.js";

const run = promisify(execFile);

test("analysis raster caps total pixels and keeps a usable artwork area", () => {
  const spec = analysisRaster(948, 1659, 600 / 1659);
  assert.ok(spec.width * spec.height <= 4_000_000);
  assert.ok(spec.contentWidth > 0 && spec.contentHeight > 0);
  assert.deepEqual(spec.contentRect, [spec.pad, spec.pad, spec.contentWidth, spec.contentHeight]);
  assert.throws(() => analysisRaster(948, 1659, 0.001), /Enlarge/);
});

test("cut analysis CLI writes inspectable contours for a local image", async () => {
  const dir = await mkdtemp(join(tmpdir(), "pixcut-cli-"));
  try {
    const input = join(dir, "two-stickers.png"),
      output = join(dir, "cuts"),
      png = new PNG({ width: 120, height: 100 });
    png.data.fill(255);
    for (const [x0, x1] of [[12, 36], [76, 102]])
      for (let y = 25; y < 75; y++) for (let x = x0; x < x1; x++) {
        const i = (y * png.width + x) * 4;
        png.data[i] = 50; png.data[i + 1] = 35; png.data[i + 2] = 160;
      }
    await writeFile(input, PNG.sync.write(png));
    const { stdout } = await run(process.execPath, [
      resolve("bin/analyze-cuts.js"), input, "--out", output,
      "--set", "mode=white", "--set", "join=0",
    ], { cwd: resolve("."), timeout: 30_000 });
    const summary = JSON.parse(await readFile(join(output, "summary.json"), "utf8")),
      analysis = JSON.parse(await readFile(join(output, "analysis.json"), "utf8")),
      overlay = PNG.sync.read(await readFile(join(output, "overlay.png")));
    assert.equal(JSON.parse(stdout).groups, 2);
    assert.equal(summary.groups.length, 2);
    assert.equal(analysis.groups.length, 2);
    assert.deepEqual(summary.issues, []);
    assert.equal(overlay.width, png.width);
    assert.equal(PNG.sync.read(await readFile(join(output, "components.png"))).width, png.width);
    assert.match(await readFile(join(output, "overlay.svg"), "utf8"), /<svg /);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
