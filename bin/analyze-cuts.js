#!/usr/bin/env node
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { basename, extname, resolve } from "node:path";
import { PNG } from "pngjs";
import jpeg from "jpeg-js";
import cvModule from "@techstark/opencv-js";
import { openCVReady } from "../src/opencv-ready.js";
import { analyze, DEFAULTS, parameters } from "../src/autocut.js";
import { analysisRaster } from "../src/analysis-raster.js";

function usage() {
  console.log(`Usage: npm run analyze:cuts -- IMAGE [--out DIR] [--fit | --scale N]
                               [--params FILE] [--set KEY=VALUE]...

Run the browser cut analyzer locally on a PNG or JPEG. The default object
scale matches importing one image on a 4×7 sticker sheet. --fit matches
"Fit artwork to page". --scale uses a specific Fabric image scale.

Writes analysis.json, summary.json, overlay.png, overlay.svg, and
components.png to DIR.
Example:
  npm run analyze:cuts -- sample.jpg --fit --out /tmp/sample-cuts
  npm run analyze:cuts -- sample.jpg --set mode=adaptive --set join=0.8`);
}

function argsOf(argv) {
  if (!argv.length || argv.includes("--help") || argv.includes("-h")) {
    usage();
    process.exit(argv.length ? 0 : 1);
  }
  const args = { input: null, out: null, fit: false, scale: null, file: null, sets: [] };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "--fit") args.fit = true;
    else if (["--out", "--scale", "--params", "--set"].includes(arg)) {
      const value = argv[++i];
      if (!value) throw Error(`${arg} needs a value.`);
      if (arg === "--out") args.out = value;
      else if (arg === "--scale") args.scale = Number(value);
      else if (arg === "--params") args.file = value;
      else args.sets.push(value);
    } else if (arg.startsWith("-")) throw Error(`Unknown option: ${arg}`);
    else if (args.input) throw Error("Give one input image.");
    else args.input = arg;
  }
  if (!args.input || (args.fit && args.scale !== null)) throw Error("Choose an image and at most one scale option.");
  if (args.scale !== null && (!Number.isFinite(args.scale) || args.scale <= 0))
    throw Error("--scale must be positive.");
  return args;
}

function decode(bytes) {
  if (bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))) {
    const image = PNG.sync.read(bytes);
    return { ...image, mime: "image/png" };
  }
  if (bytes[0] === 255 && bytes[1] === 216) {
    const image = jpeg.decode(bytes, { useTArray: true });
    return { ...image, mime: "image/jpeg" };
  }
  throw Error("The CLI supports PNG and JPEG images.");
}

function resizeRGBA(source, targetWidth, targetHeight) {
  const out = new Uint8Array(targetWidth * targetHeight * 4),
    xScale = source.width / targetWidth,
    yScale = source.height / targetHeight;
  for (let y = 0; y < targetHeight; y++) {
    const sy = Math.max(0, Math.min(source.height - 1, (y + 0.5) * yScale - 0.5)),
      y0 = Math.floor(sy), y1 = Math.min(source.height - 1, y0 + 1), fy = sy - y0;
    for (let x = 0; x < targetWidth; x++) {
      const sx = Math.max(0, Math.min(source.width - 1, (x + 0.5) * xScale - 0.5)),
        x0 = Math.floor(sx), x1 = Math.min(source.width - 1, x0 + 1), fx = sx - x0,
        output = (y * targetWidth + x) * 4,
        samples = [
          [(y0 * source.width + x0) * 4, (1 - fx) * (1 - fy)],
          [(y0 * source.width + x1) * 4, fx * (1 - fy)],
          [(y1 * source.width + x0) * 4, (1 - fx) * fy],
          [(y1 * source.width + x1) * 4, fx * fy],
        ];
      let alpha = 0;
      for (const [i, weight] of samples) alpha += source.data[i + 3] * weight;
      out[output + 3] = Math.round(alpha);
      for (let channel = 0; channel < 3; channel++) {
        let value = 0;
        for (const [i, weight] of samples)
          value += source.data[i + channel] * source.data[i + 3] * weight;
        out[output + channel] = alpha ? Math.round(value / alpha) : 0;
      }
    }
  }
  return out;
}

function rasterize(source, spec) {
  const rgba = new Uint8Array(spec.width * spec.height * 4),
    content = resizeRGBA(source, spec.contentWidth, spec.contentHeight);
  for (let y = 0; y < spec.contentHeight; y++)
    rgba.set(content.subarray(y * spec.contentWidth * 4, (y + 1) * spec.contentWidth * 4),
      ((y + spec.pad) * spec.width + spec.pad) * 4);
  return rgba;
}

const colorFor = (index) => {
  const hue = (index * 137.508) % 360,
    chroma = 0.72, light = 0.38,
    x = chroma * (1 - Math.abs((hue / 60) % 2 - 1)),
    m = light - chroma / 2,
    sectors = [[chroma, x, 0], [x, chroma, 0], [0, chroma, x],
      [0, x, chroma], [x, 0, chroma], [chroma, 0, x]];
  return sectors[Math.floor(hue / 60)].map((v) => Math.round(255 * (v + m)));
};
const hex = (rgb) => `#${rgb.map((v) => v.toString(16).padStart(2, "0")).join("")}`;

function overlayPNG(source, result, spec) {
  const png = new PNG({ width: source.width, height: source.height });
  for (let i = 0; i < source.width * source.height; i++) {
    const a = source.data[i * 4 + 3] / 255;
    for (let c = 0; c < 3; c++) png.data[i * 4 + c] = Math.round(source.data[i * 4 + c] * a + 255 * (1 - a));
    png.data[i * 4 + 3] = 255;
  }
  const spot = (x, y, radius, rgb) => {
    x = Math.round(x); y = Math.round(y);
    for (let dy = -radius; dy <= radius; dy++) for (let dx = -radius; dx <= radius; dx++) {
      if (dx * dx + dy * dy > radius * radius) continue;
      const xx = x + dx, yy = y + dy;
      if (xx < 0 || yy < 0 || xx >= source.width || yy >= source.height) continue;
      const i = (yy * source.width + xx) * 4;
      png.data[i] = rgb[0]; png.data[i + 1] = rgb[1]; png.data[i + 2] = rgb[2];
    }
  };
  const paths = result.groups.flatMap((group, g) => group.paths.map((path) => ({ path, g })));
  for (const radius of [3, 1]) for (const { path, g } of paths) {
    const color = radius === 3 ? [255, 255, 255] : colorFor(g);
    for (let i = 0; i < path.length; i++) {
      const a = path[i], b = path[(i + 1) % path.length],
        ax = (a[0] - spec.pad) / spec.factor, ay = (a[1] - spec.pad) / spec.factor,
        bx = (b[0] - spec.pad) / spec.factor, by = (b[1] - spec.pad) / spec.factor,
        steps = Math.max(1, Math.ceil(Math.hypot(bx - ax, by - ay) * 2));
      for (let step = 0; step <= steps; step++)
        spot(ax + (bx - ax) * step / steps, ay + (by - ay) * step / steps, radius, color);
    }
  }
  return PNG.sync.write(png);
}

function componentsPNG(source, result, spec) {
  const png = new PNG({ width: source.width, height: source.height }),
    labels = new Int32Array(spec.width * spec.height),
    used = new Set(result.groups.flatMap((g) => g.componentIds));
  for (const component of result.components)
    for (const [start, length] of component.runs)
      labels.fill(component.id, start, start + length);
  for (let y = 0; y < source.height; y++) for (let x = 0; x < source.width; x++) {
    const rx = Math.min(spec.contentWidth - 1, Math.round((x + 0.5) * spec.factor - 0.5)) + spec.pad,
      ry = Math.min(spec.contentHeight - 1, Math.round((y + 0.5) * spec.factor - 0.5)) + spec.pad,
      id = labels[ry * spec.width + rx],
      target = (y * source.width + x) * 4,
      color = id ? used.has(id) ? colorFor(id) : [220, 35, 35] : [255, 255, 255];
    for (let c = 0; c < 3; c++)
      png.data[target + c] = Math.round(source.data[target + c] * 0.35 + color[c] * 0.65);
    png.data[target + 3] = 255;
  }
  return PNG.sync.write(png);
}

function overlaySVG(source, originalBytes, result, spec) {
  const project = ([x, y]) => `${((x - spec.pad) / spec.factor).toFixed(2)},${((y - spec.pad) / spec.factor).toFixed(2)}`;
  const paths = result.groups.flatMap((group, g) => group.paths.map((path) => {
    const d = `M ${path.map(project).join(" L ")} Z`, color = hex(colorFor(g)),
      x = path.reduce((n, p) => n + p[0], 0) / path.length,
      y = path.reduce((n, p) => n + p[1], 0) / path.length;
    return `<path d="${d}" fill="none" stroke="white" stroke-width="6"/>\n` +
      `<path d="${d}" fill="none" stroke="${color}" stroke-width="2"/>\n` +
      `<text x="${((x - spec.pad) / spec.factor).toFixed(1)}" y="${((y - spec.pad) / spec.factor).toFixed(1)}" ` +
      `paint-order="stroke" stroke="white" stroke-width="4" fill="${color}" ` +
      `font-family="sans-serif" font-size="17" font-weight="bold" text-anchor="middle">${g + 1}</text>`;
  })).join("\n");
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${source.width}" height="${source.height}" ` +
    `viewBox="0 0 ${source.width} ${source.height}">\n` +
    `<image width="${source.width}" height="${source.height}" href="data:${source.mime};base64,${originalBytes.toString("base64")}"/>\n` +
    `${paths}\n</svg>\n`;
}

async function main() {
  const args = argsOf(process.argv.slice(2)),
    input = resolve(args.input),
    bytes = await readFile(input),
    source = decode(bytes);
  if (source.width * source.height > 40_000_000) throw Error("Image exceeds the editor's 40 megapixel limit.");
  const scale = args.scale ?? (args.fit
    ? Math.min(1080 / source.width, 1980 / source.height)
    : Math.min(450 / source.width, 600 / source.height));
  const spec = analysisRaster(source.width, source.height, scale),
    overrides = args.file ? JSON.parse(await readFile(resolve(args.file), "utf8")) : {};
  if (!overrides || typeof overrides !== "object" || Array.isArray(overrides) ||
      Object.keys(overrides).some((key) => !(key in DEFAULTS)))
    throw Error("Parameter file must be an object with known cut parameters.");
  for (const entry of args.sets) {
    const at = entry.indexOf("="), key = entry.slice(0, at), raw = entry.slice(at + 1);
    if (at < 1 || !(key in DEFAULTS)) throw Error(`Invalid parameter: ${entry}`);
    overrides[key] = typeof DEFAULTS[key] === "number" ? Number(raw)
      : typeof DEFAULTS[key] === "boolean" ? raw === "true" ? true : raw === "false" ? false : null
      : raw;
  }
  if (typeof overrides.keepSmall !== "undefined" && typeof overrides.keepSmall !== "boolean")
    throw Error("keepSmall must be true or false.");
  const params = parameters({ ...DEFAULTS, ...overrides }),
    rgba = rasterize(source, spec),
    { cv } = await openCVReady(cvModule),
    start = performance.now(),
    result = analyze(cv, {
      width: spec.width, height: spec.height, pxPerMM: spec.pxPerMM,
      contentRect: spec.contentRect, rgba, params,
    }, (step) => process.stderr.write(`${step}\n`));
  const used = new Set(result.groups.flatMap((g) => g.componentIds)),
    summary = {
      input, objectScale: scale, raster: spec, mode: result.mode,
      seconds: Number(((performance.now() - start) / 1000).toFixed(3)),
      components: result.components.length,
      excludedComponentIds: result.components.filter((c) => !used.has(c.id)).map((c) => c.id),
      groups: result.groups.map((g, i) => ({
        number: i + 1, componentIds: g.componentIds, outlines: g.paths.length,
        vertices: g.paths.map((path) => path.length), clippedPixels: g.clipped,
      })),
      issues: result.issues,
    };
  const out = resolve(args.out || `cut-analysis-${basename(input, extname(input))}`);
  await mkdir(out, { recursive: true });
  await Promise.all([
    writeFile(resolve(out, "analysis.json"), JSON.stringify(result)),
    writeFile(resolve(out, "summary.json"), JSON.stringify(summary, null, 2) + "\n"),
    writeFile(resolve(out, "overlay.png"), overlayPNG(source, result, spec)),
    writeFile(resolve(out, "overlay.svg"), overlaySVG(source, bytes, result, spec)),
    writeFile(resolve(out, "components.png"), componentsPNG(source, result, spec)),
  ]);
  console.log(JSON.stringify({ out, mode: summary.mode, components: summary.components,
    groups: summary.groups.length, excluded: summary.excludedComponentIds.length,
    issues: summary.issues.length, seconds: summary.seconds }));
}

main().catch((error) => {
  console.error(error.message || String(error));
  process.exitCode = 1;
});
