import { PX_PER_MM } from "./geometry.js";

// Use the same analysis grid in the editor and the development CLI.
export function analysisRaster(width, height, objectScale) {
  if (![width, height, objectScale].every((n) => Number.isFinite(n) && n > 0))
    throw Error("Invalid artwork dimensions or scale.");
  let factor = Math.min(1, 1700 / Math.max(width, height)),
    pxPerMM = (PX_PER_MM * factor) / objectScale,
    pad = Math.ceil(pxPerMM * 12),
    contentWidth = Math.round(width * factor),
    contentHeight = Math.round(height * factor),
    rasterWidth = contentWidth + 2 * pad,
    rasterHeight = contentHeight + 2 * pad;
  if (pxPerMM > 100 || contentWidth < 1 || contentHeight < 1)
    throw Error("Enlarge the artwork before analyzing it.");
  while (rasterWidth * rasterHeight > 4_000_000) {
    factor *= Math.min(0.99, Math.sqrt(4_000_000 / (rasterWidth * rasterHeight)));
    pxPerMM = (PX_PER_MM * factor) / objectScale;
    pad = Math.ceil(pxPerMM * 12);
    contentWidth = Math.round(width * factor);
    contentHeight = Math.round(height * factor);
    rasterWidth = contentWidth + 2 * pad;
    rasterHeight = contentHeight + 2 * pad;
  }
  if (contentWidth < 1 || contentHeight < 1)
    throw Error("Enlarge the artwork before analyzing it.");
  return {
    factor, pad, pxPerMM,
    width: rasterWidth, height: rasterHeight,
    contentWidth, contentHeight,
    contentRect: [pad, pad, contentWidth, contentHeight],
  };
}
