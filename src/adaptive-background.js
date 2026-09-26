// Estimate the slowly varying color of a printed sheet from its exposed edges
// and from matching pixels between the artwork. OpenCV's 8-bit Lab space keeps
// lightness separate from the pastel color changes.
export function adaptiveMask(cv, rgba, width, height, contentRect, alpha) {
  const [left, top, contentWidth, contentHeight] = contentRect || [0, 0, width, height];
  if (
    ![left, top, contentWidth, contentHeight].every(Number.isInteger) ||
    left < 0 || top < 0 || contentWidth < 1 || contentHeight < 1 ||
    left + contentWidth > width || top + contentHeight > height
  ) throw Error("Invalid image content area.");

  const source = cv.matFromArray(height, width, cv.CV_8UC4, rgba);
  const rgb = new cv.Mat(), lab = new cv.Mat();
  try {
    cv.cvtColor(source, rgb, cv.COLOR_RGBA2RGB);
    cv.cvtColor(rgb, lab, cv.COLOR_RGB2Lab);
    const colors = lab.data, edge = new Float32Array(contentHeight * 6),
      radius = Math.max(8, Math.min(64, Math.round(contentHeight / 24))),
      borderWidth = Math.min(5, contentWidth);
    for (let y = 0; y < contentHeight; y++) {
      for (let side = 0; side < 2; side++) {
        let samples = 0;
        for (let dx = 0; dx < borderWidth; dx++) {
          const x = side ? left + contentWidth - 1 - dx : left + dx,
            pixel = (top + y) * width + x;
          if (rgba[pixel * 4 + 3] < alpha) continue;
          for (let c = 0; c < 3; c++) edge[y * 6 + side * 3 + c] += colors[pixel * 3 + c];
          samples++;
        }
        for (let c = 0; c < 3; c++) edge[y * 6 + side * 3 + c] =
          samples ? edge[y * 6 + side * 3 + c] / samples : (c ? 128 : 255);
      }
    }
    const smoothed = new Float32Array(edge.length);
    for (let c = 0; c < 6; c++) {
      const prefix = new Float64Array(contentHeight + 1);
      for (let y = 0; y < contentHeight; y++) prefix[y + 1] = prefix[y] + edge[y * 6 + c];
      for (let y = 0; y < contentHeight; y++) {
        const lo = Math.max(0, y - radius), hi = Math.min(contentHeight, y + radius + 1);
        smoothed[y * 6 + c] = (prefix[hi] - prefix[lo]) / (hi - lo);
      }
    }

    const tileSize = 48, columns = Math.ceil(contentWidth / tileSize),
      rows = Math.ceil(contentHeight / tileSize), cells = columns * rows,
      sums = new Float64Array(cells * 3), counts = new Uint32Array(cells),
      field = new Float32Array(cells * 3), fixed = new Uint8Array(cells);
    const edgeColor = (x, y, c) => {
      const t = contentWidth === 1 ? 0 : x / (contentWidth - 1), row = y * 6 + c;
      return smoothed[row] * (1 - t) + smoothed[row + 3] * t;
    };
    for (let y = 0; y < contentHeight; y++) for (let x = 0; x < contentWidth; x++) {
      const pixel = (top + y) * width + left + x;
      if (rgba[pixel * 4 + 3] < alpha) continue;
      const i = pixel * 3, l = edgeColor(x, y, 0), a = edgeColor(x, y, 1), b = edgeColor(x, y, 2),
        dl = colors[i] - l, da = colors[i + 1] - a, db = colors[i + 2] - b;
      if (dl * dl + da * da + db * db > 17 * 17 || (colors[i] > 250 && l < 245)) continue;
      const cell = Math.floor(y / tileSize) * columns + Math.floor(x / tileSize);
      counts[cell]++;
      for (let c = 0; c < 3; c++) sums[cell * 3 + c] += colors[i + c];
    }
    for (let row = 0; row < rows; row++) for (let col = 0; col < columns; col++) {
      const cell = row * columns + col, x = Math.min(contentWidth - 1, col * tileSize + tileSize / 2),
        y = Math.min(contentHeight - 1, row * tileSize + tileSize / 2);
      const cellArea = Math.min(tileSize, contentWidth - col * tileSize) *
        Math.min(tileSize, contentHeight - row * tileSize);
      fixed[cell] = counts[cell] >= Math.max(16, cellArea * 0.06) ? 1 : 0;
      for (let c = 0; c < 3; c++) field[cell * 3 + c] = fixed[cell]
        ? sums[cell * 3 + c] / counts[cell] : edgeColor(x, y, c);
    }
    // Interpolate through cells hidden by large stickers, holding observed
    // background cells fixed and retaining a weak edge-model prior.
    for (let pass = 0; pass < 48; pass++) for (let row = 0; row < rows; row++)
      for (let col = 0; col < columns; col++) {
        const cell = row * columns + col;
        if (fixed[cell]) continue;
        const neighbors = [
          col ? cell - 1 : -1, col + 1 < columns ? cell + 1 : -1,
          row ? cell - columns : -1, row + 1 < rows ? cell + columns : -1,
        ].filter((n) => n >= 0);
        for (let c = 0; c < 3; c++) {
          const prior = edgeColor(Math.min(contentWidth - 1, col * tileSize + tileSize / 2),
            Math.min(contentHeight - 1, row * tileSize + tileSize / 2), c);
          field[cell * 3 + c] = (prior * 0.15 + neighbors.reduce((n, cell) => n + field[cell * 3 + c], 0)) /
            (neighbors.length + 0.15);
        }
      }

    const result = new Uint8Array(width * height), differenceMap = new Float32Array(width * height),
      rim = new Uint8Array(width * height);
    for (let y = 0; y < contentHeight; y++) for (let x = 0; x < contentWidth; x++) {
      const pixel = (top + y) * width + left + x;
      if (rgba[pixel * 4 + 3] < alpha) continue;
      const gx = Math.max(0, Math.min(columns - 1, (x + 0.5) / tileSize - 0.5)),
        gy = Math.max(0, Math.min(rows - 1, (y + 0.5) / tileSize - 0.5)),
        x0 = Math.floor(gx), y0 = Math.floor(gy), x1 = Math.min(columns - 1, x0 + 1),
        y1 = Math.min(rows - 1, y0 + 1), tx = gx - x0, ty = gy - y0,
        i = pixel * 3;
      const local = (c) => {
        const topValue = field[(y0 * columns + x0) * 3 + c] * (1 - tx) +
          field[(y0 * columns + x1) * 3 + c] * tx;
        const bottomValue = field[(y1 * columns + x0) * 3 + c] * (1 - tx) +
          field[(y1 * columns + x1) * 3 + c] * tx;
        return topValue * (1 - ty) + bottomValue * ty;
      };
      const l = local(0), a = local(1), b = local(2),
        dl = colors[i] - l, da = colors[i + 1] - a, db = colors[i + 2] - b,
        difference = dl * dl + da * da + db * db,
        // A bright neutral rim is part of the sticker even when the nearby
        // pastel is too similar for the general color-distance test.
        whiteRim = colors[i] >= 248 && colors[i] > l + 8 &&
          Math.abs(colors[i + 1] - 128) + Math.abs(colors[i + 2] - 128) < 18;
      differenceMap[pixel] = difference;
      rim[pixel] = whiteRim ? 1 : 0;
      if (difference > 22 * 22 || whiteRim) result[pixel] = 255;
    }
    // Background clouds may differ more than the threshold from the local
    // estimate. Grow the exposed backdrop through smooth, similarly colored
    // pixels, stopping at the bright sticker rim and abrupt artwork edges.
    const visited = new Uint8Array(width * height), queue = new Int32Array(width * height);
    let head = 0, tail = 0;
    const visit = (pixel) => {
      if (visited[pixel] || rim[pixel] || differenceMap[pixel] > 42 * 42 ||
          colors[pixel * 3] < 205 || rgba[pixel * 4 + 3] < alpha) return;
      visited[pixel] = 1; result[pixel] = 0; queue[tail++] = pixel;
    };
    for (let x = left; x < left + contentWidth; x++) {
      visit(top * width + x); visit((top + contentHeight - 1) * width + x);
    }
    for (let y = top; y < top + contentHeight; y++) {
      visit(y * width + left); visit(y * width + left + contentWidth - 1);
    }
    while (head < tail) {
      const pixel = queue[head++], x = pixel % width, y = (pixel - x) / width,
        neighbors = [
          x > left ? pixel - 1 : -1, x < left + contentWidth - 1 ? pixel + 1 : -1,
          y > top ? pixel - width : -1, y < top + contentHeight - 1 ? pixel + width : -1,
        ];
      for (const neighbor of neighbors) {
        if (neighbor < 0 || visited[neighbor]) continue;
        const a = pixel * 3, b = neighbor * 3,
          dl = colors[a] - colors[b], da = colors[a + 1] - colors[b + 1],
          db = colors[a + 2] - colors[b + 2];
        if (dl * dl + da * da + db * db <= 8 * 8) visit(neighbor);
      }
    }
    return result;
  } finally {
    source.delete(); rgb.delete(); lab.delete();
  }
}
