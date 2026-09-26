// Pure orchestration around OpenCV; shared by the worker and regression tests.
import { adaptiveMask } from "./adaptive-background.js";
import { splitTouchingComponent } from "./touching-split.js";
export const DEFAULTS = Object.freeze({
  mode: "auto",
  saturation: 35,
  darkness: 175,
  alpha: 128,
  background: "#ffffff",
  tolerance: 35,
  minArea: 0.15,
  join: 1.1,
  keepSmall: false,
  smallArea: 2,
  border: 0.6,
  bridge: 1.1,
  smooth: 0.4,
  gap: 0.1,
  detail: 0.06,
});
export function parameters(input = {}) {
  const p = { ...DEFAULTS, ...input };
  for (const [key, min, max] of [
    ["saturation", 0, 255],
    ["darkness", 0, 255],
    ["alpha", 1, 255],
    ["tolerance", 1, 255],
    ["minArea", 0, 10],
    ["join", 0, 5],
    ["smallArea", 0, 50],
    ["border", 0, 5],
    ["bridge", 0, 5],
    ["smooth", 0, 3],
    ["gap", 0, 3],
    ["detail", 0, 1],
  ]) {
    if (!Number.isFinite(p[key]) || p[key] < min || p[key] > max)
      throw Error(`Invalid ${key} parameter.`);
  }
  if (
    !["auto", "alpha", "white", "color", "adaptive"].includes(p.mode) ||
    !/^#[a-f\d]{6}$/i.test(p.background)
  )
    throw Error("Invalid background settings.");
  return p;
}
export function maskPixels(rgba, p) {
  p = parameters(p);
  const result = new Uint8Array(rgba.length / 4);
  let transparent = 0;
  for (let i = 3; i < rgba.length; i += 4) if (rgba[i] < p.alpha) transparent++;
  const mode =
    p.mode === "auto"
      ? transparent > result.length * 0.005
        ? "alpha"
        : "white"
      : p.mode;
  if (mode === "adaptive") throw Error("Adaptive detection needs image dimensions.");
  const bg = [1, 3, 5].map((i) => parseInt(p.background.slice(i, i + 2), 16));
  for (let i = 0; i < result.length; i++) {
    const a = rgba[i * 4 + 3];
    if (a < p.alpha) continue;
    const rgb = [0, 1, 2].map((k) => rgba[i * 4 + k]),
      v = Math.max(...rgb),
      s = v ? (255 * (v - Math.min(...rgb))) / v : 0;
    result[i] =
      mode === "alpha" ||
      (mode === "white"
        ? s > p.saturation || v < p.darkness
        : Math.hypot(...rgb.map((v, k) => v - bg[k])) > p.tolerance)
        ? 255
        : 0;
  }
  return { mask: result, mode };
}
function automaticMode(rgba, width, height, contentRect, alpha) {
  const [x0, y0, cw, ch] = contentRect || [0, 0, width, height];
  if (![x0, y0, cw, ch].every(Number.isInteger) || x0 < 0 || y0 < 0 ||
      cw < 1 || ch < 1 || x0 + cw > width || y0 + ch > height)
    throw Error("Invalid image content area.");
  let transparent = 0;
  for (let y = y0; y < y0 + ch; y++) for (let x = x0; x < x0 + cw; x++)
    if (rgba[(y * width + x) * 4 + 3] < alpha) transparent++;
  if (transparent > cw * ch * 0.005) return "alpha";
  let samples = 0, white = 0;
  const sample = (x, y) => {
    const i = (y * width + x) * 4;
    samples++;
    if (rgba[i] >= 245 && rgba[i + 1] >= 245 && rgba[i + 2] >= 245) white++;
  };
  const step = Math.max(1, Math.floor((cw + ch) / 500));
  for (let x = x0; x < x0 + cw; x += step) {
    sample(x, y0); sample(x, y0 + ch - 1);
  }
  for (let y = y0; y < y0 + ch; y += step) {
    sample(x0, y); sample(x0 + cw - 1, y);
  }
  return white / samples >= 0.9 ? "white" : "adaptive";
}
export function groupKey(ids) {
  return [...ids].sort((a, b) => a - b).join(",");
}
export function mergeGroups(groups, indices) {
  const selected = new Set(indices);
  if (selected.size < 2) throw Error("Select at least two groups to merge.");
  const parts = groups.filter((_, i) => selected.has(i));
  if (parts.some((g) => g.override))
    throw Error("Reset locked outlines before merging their groups.");
  return [
    ...groups.filter((_, i) => !selected.has(i)),
    {
      componentIds: parts.flatMap((g) => g.componentIds),
      seedPaths: parts.flatMap((g) => g.seedPaths || []),
    },
  ];
}
export function splitGroup(groups, index) {
  const g = groups[index];
  if (!g || g.componentIds.length < 2)
    throw Error(
      "This group has only one connected component. Use a manual outline for touching artwork.",
    );
  if (g.override)
    throw Error("Reset the locked outline before splitting its group.");
  return [
    ...groups.filter((_, i) => i !== index),
    ...g.componentIds.map((id) => ({ componentIds: [id] })),
  ];
}
export function moveComponents(groups, ids, target) {
  const move = new Set(ids);
  if (!move.size || (!groups[target] && target !== -1))
    throw Error("Select components and a destination group.");
  if (
    groups.some(
      (g, i) =>
        (i === target || g.componentIds.some((id) => move.has(id))) &&
        g.override,
    )
  )
    throw Error("Reset affected locked outlines before assigning components.");
  const result = groups
    .map((g, i) => ({
      ...g,
      componentIds:
        i === target
          ? [...new Set([...g.componentIds, ...ids])]
          : g.componentIds.filter((id) => !move.has(id)),
      seedPaths: undefined,
    }))
    .filter((g) => g.componentIds.length);
  if (target === -1) result.push({ componentIds: ids });
  return result;
}

export function analyze(cv, request, progress = () => {}) {
  const p = parameters(request.params),
    { width: w, height: h, pxPerMM: mm } = request;
  if (
    !Number.isInteger(w) ||
    !Number.isInteger(h) ||
    w < 1 ||
    h < 1 ||
    w * h > 4_000_000 ||
    !Number.isFinite(mm) ||
    mm <= 0 ||
    mm > 100
  )
    throw Error("Invalid analysis dimensions.");
  const mats = [];
  const own = (m) => {
    mats.push(m);
    return m;
  };
  const empty = (width = w, height = h) =>
    own(cv.Mat.zeros(height, width, cv.CV_8UC1));
  const kernel = (r) =>
    own(
      cv.getStructuringElement(
        cv.MORPH_ELLIPSE,
        new cv.Size(2 * r + 1, 2 * r + 1),
      ),
    );
  const morph = (src, op, r) => {
    if (r > 0) cv.morphologyEx(src, src, op, kernel(r));
  };
  const dilate = (src, r) => {
    const out = own(new cv.Mat());
    if (r > 0) cv.dilate(src, out, kernel(r));
    else src.copyTo(out);
    return out;
  };
  const fill = (paths, target, value = 255) => {
    const vectors = new cv.MatVector();
    try {
      for (const path of paths) {
        const m = cv.matFromArray(
          path.length,
          1,
          cv.CV_32SC2,
          path.flatMap((pt) => pt.map(Math.round)),
        );
        vectors.push_back(m);
        m.delete();
      }
      cv.fillPoly(target, vectors, new cv.Scalar(value));
    } finally {
      vectors.delete();
    }
  };
  const contours = (mask, epsilon) => {
    const found = new cv.MatVector(),
      hierarchy = new cv.Mat(),
      out = [];
    try {
      cv.findContours(
        mask,
        found,
        hierarchy,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE,
      );
      for (let i = 0; i < found.size(); i++) {
        const raw = found.get(i),
          simple = new cv.Mat();
        try {
          cv.approxPolyDP(raw, simple, epsilon, true);
          const path = [];
          for (let j = 0; j < simple.data32S.length; j += 2)
            path.push([simple.data32S[j], simple.data32S[j + 1]]);
          if (path.length >= 3) out.push(path);
        } finally {
          raw.delete();
          simple.delete();
        }
      }
    } finally {
      found.delete();
      hierarchy.delete();
    }
    return out;
  };
  try {
    progress("Detecting artwork");
    let components = request.components,
      mode = request.mode,
      groups = request.groups;
    if (!components) {
      if (request.rgba?.length !== w * h * 4)
        throw Error("Invalid image data.");
      mode = p.mode === "auto"
        ? automaticMode(request.rgba, w, h, request.contentRect, p.alpha)
        : p.mode;
      const ink = empty();
      ink.data.set(mode === "adaptive"
        ? adaptiveMask(cv, request.rgba, w, h, request.contentRect, p.alpha)
        : maskPixels(request.rgba, { ...p, mode }).mask);
      const labels = own(new cv.Mat()),
        stats = own(new cv.Mat()),
        centers = own(new cv.Mat());
      const count = cv.connectedComponentsWithStats(
        ink,
        labels,
        stats,
        centers,
        8,
        cv.CV_32S,
      );
      const byLabel = new Map();
      components = [];
      for (let id = 1; id < count; id++) {
        const row = id * 5,
          area = stats.data32S[row + 4];
        if (area < Math.max(1, p.minArea * mm * mm)) continue;
        const c = {
          id,
          area,
          bounds: Array.from(stats.data32S.slice(row, row + 4)),
          center: Array.from(centers.data64F.slice(id * 2, id * 2 + 2)),
          runs: [],
        };
        components.push(c);
        byLabel.set(id, c);
      }
      if (!components.length)
        throw Error(
          "No artwork detected. Increase sensitivity or change background mode.",
        );
      if (components.length > 2000)
        throw Error(
          "More than 2,000 components. Increase minimum feature area or reduce sensitivity.",
        );
      for (let i = 0; i < w * h; ) {
        const id = labels.data32S[i],
          start = i;
        while (i < w * h && labels.data32S[i] === id) i++;
        byLabel.get(id)?.runs.push([start, i - start]);
      }
      if (request.seedPaths?.length) {
        if (request.seedPaths.length > 250)
          throw Error("Too many seed regions.");
        groups = request.seedPaths.map((path) => ({
          componentIds: [],
          seedPaths: [path],
        }));
        const assignments = components.map(() => ({
          best: -1,
          overlap: -1,
          nearest: Infinity,
        }));
        for (let g = 0; g < groups.length; g++) {
          const region = empty(),
            inv = empty(),
            distanceMap = own(new cv.Mat());
          fill(groups[g].seedPaths, region);
          cv.bitwise_not(region, inv);
          cv.distanceTransform(
            inv,
            distanceMap,
            cv.DIST_L2,
            cv.DIST_MASK_PRECISE,
          );
          components.forEach((c, index) => {
            let hits = 0,
              distance = Infinity;
            for (const [start, len] of c.runs)
              for (let i = start; i < start + len; i++) {
                if (region.data[i]) hits++;
                distance = Math.min(distance, distanceMap.data32F[i]);
              }
            const a = assignments[index];
            if (
              hits > a.overlap ||
              (hits === a.overlap && distance < a.nearest)
            )
              assignments[index] = {
                best: g,
                overlap: hits,
                nearest: distance,
              };
          });
          for (const m of [region, inv, distanceMap]) {
            m.delete();
            mats.splice(mats.indexOf(m), 1);
          }
        }
        components.forEach((c, i) =>
          groups[assignments[i].best].componentIds.push(c.id),
        );
        groups = groups.filter((g) => g.componentIds.length);
      } else if (mode === "adaptive") {
        // Tiny background glints must not form transitive bridges between
        // stickers. Use substantial components as independent anchors, then
        // attach each smaller piece to at most one nearby anchor.
        const anchorArea = Math.max(p.smallArea, p.minArea) * mm * mm,
          anchors = components.filter((c) => c.area >= anchorArea);
        if (!anchors.length) anchors.push(components.reduce((a, b) => a.area > b.area ? a : b));
        const anchorIds = new Set(anchors.map((c) => c.id)),
          host = new Map(anchors.map((c) => [c.id, c.id])),
          nearby = new Map(), joinPixels = p.join * mm;
        for (const anchor of anchors) {
          const [x0, y0, rw, rh] = anchor.bounds,
            shape = cv.Mat.zeros(rh, rw, cv.CV_8UC1),
            found = new cv.MatVector(), hierarchy = new cv.Mat();
          try {
            for (const [start, length] of anchor.runs)
              for (let at = start; at < start + length; at++) {
                const x = at % w, y = (at - x) / w;
                shape.data[(y - y0) * rw + x - x0] = 255;
              }
            cv.findContours(shape, found, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
            for (const piece of components) {
              if (piece.id === anchor.id ||
                  (piece.area >= anchorArea && piece.area >= anchor.area)) continue;
              const [x, y] = piece.center;
              if (x < x0 - joinPixels || x > x0 + rw + joinPixels ||
                  y < y0 - joinPixels || y > y0 + rh + joinPixels) continue;
              let signedDistance = -Infinity;
              for (let i = 0; i < found.size(); i++) {
                const path = found.get(i);
                try {
                  signedDistance = Math.max(signedDistance,
                    cv.pointPolygonTest(path, new cv.Point(x - x0, y - y0), true));
                } finally { path.delete(); }
              }
              if (piece.area >= anchorArea) {
                if (signedDistance < 0) continue;
                const prior = host.get(piece.id), previous = components.find((c) => c.id === prior);
                if (prior === piece.id || anchor.area < previous.area) host.set(piece.id, anchor.id);
              } else if (signedDistance >= -joinPixels) {
                const prior = nearby.get(piece.id);
                if (!prior || signedDistance > prior.distance)
                  nearby.set(piece.id, { anchorId: anchor.id, distance: signedDistance });
              }
            }
          } finally {
            shape.delete(); found.delete(); hierarchy.delete();
          }
        }
        const root = (id) => {
          while (host.get(id) !== id) id = host.get(id);
          return id;
        };
        const byAnchor = new Map();
        for (const anchor of anchors) {
          const id = root(anchor.id);
          if (!byAnchor.has(id)) byAnchor.set(id, []);
          byAnchor.get(id).push(anchor.id);
        }
        for (const piece of components.filter((c) => !anchorIds.has(c.id))) {
          const match = nearby.get(piece.id);
          if (p.keepSmall || !match) continue;
          byAnchor.get(root(match.anchorId)).push(piece.id);
        }
        groups = [...byAnchor.values()].map((componentIds) => ({ componentIds }));
        if (p.keepSmall) groups.push(...components.filter((c) => !anchorIds.has(c.id))
          .map((c) => ({ componentIds: [c.id] })));
      } else {
        const joinMask = empty();
        for (const c of components)
          if (!p.keepSmall || c.area >= p.smallArea * mm * mm)
            for (const [s, n] of c.runs) joinMask.data.fill(255, s, s + n);
        const joined = dilate(joinMask, Math.round((p.join * mm) / 2)),
          joinedLabels = own(new cv.Mat());
        cv.connectedComponents(joined, joinedLabels, 8, cv.CV_32S);
        const sets = new Map();
        for (const c of components) {
          const small = p.keepSmall && c.area < p.smallArea * mm * mm,
            key = small
              ? `small-${c.id}`
              : `join-${joinedLabels.data32S[c.runs[0][0]]}`;
          if (!sets.has(key)) sets.set(key, []);
          sets.get(key).push(c.id);
        }
        groups = [...sets.values()].map((componentIds) => ({ componentIds }));
      }
    }
    if (request.splitTouching) {
      progress("Separating touching stickers");
      ({ components, groups } = splitTouchingComponent(
        cv, w, h, components, groups,
        request.splitTouching.componentId, request.splitTouching.points,
        Math.max(2, Math.ceil((p.gap * mm) / 2) + 4),
      ));
    }
    if (request.groupingOnly) return { groups, components };
    if (!groups?.length || groups.length > 250)
      throw Error(
        "Need 1–250 sticker groups. Increase joining, reduce the small-decoration threshold, or uncheck “Keep small decorations separate”.",
      );
    const lookup = new Map(components.map((c) => [c.id, c]));
    const seen = new Set();
    for (const g of groups)
      for (const id of g.componentIds) {
        if (!lookup.has(id) || seen.has(id))
          throw Error("Invalid or duplicate component assignment.");
        seen.add(id);
      }
    const artFor = (g) => {
      const art = empty();
      for (const id of g.componentIds)
        for (const [s, n] of lookup.get(id).runs) art.data.fill(255, s, s + n);
      return art;
    };
    const regionFor = (group) => {
      const bounds = group.componentIds.map((id) => lookup.get(id).bounds),
        seedPoints = (group.seedPaths || []).flat(),
        padding = Math.ceil((p.border + p.bridge + p.smooth + 2) * mm) + 3;
      const x0 = Math.max(0, Math.floor(Math.min(
          ...bounds.map((b) => b[0]), ...seedPoints.map((pt) => pt[0]),
        ) - padding)),
        y0 = Math.max(0, Math.floor(Math.min(
          ...bounds.map((b) => b[1]), ...seedPoints.map((pt) => pt[1]),
        ) - padding)),
        x1 = Math.min(w, Math.ceil(Math.max(
          ...bounds.map((b) => b[0] + b[2]), ...seedPoints.map((pt) => pt[0]),
        ) + padding)),
        y1 = Math.min(h, Math.ceil(Math.max(
          ...bounds.map((b) => b[1] + b[3]), ...seedPoints.map((pt) => pt[1]),
        ) + padding));
      return { x0, y0, rw: x1 - x0, rh: y1 - y0 };
    };
    const regions = groups.map(regionFor);
    progress("Separating neighboring stickers");
    const owner = new Int16Array(w * h).fill(-1),
      best = new Float32Array(w * h).fill(Infinity);
    // A cut can only occupy its padded group region. Compute the nearest
    // artwork there instead of transforming the entire sheet for every group.
    for (let g = 0; g < groups.length; g++) {
      const { x0, y0, rw, rh } = regions[g];
      const art = artFor(groups[g]),
        localArt = own(art.roi(new cv.Rect(x0, y0, rw, rh))),
        inv = empty(rw, rh),
        dist = own(new cv.Mat());
      cv.bitwise_not(localArt, inv);
      cv.distanceTransform(inv, dist, cv.DIST_L2, cv.DIST_MASK_PRECISE);
      for (let y = 0; y < rh; y++) for (let x = 0; x < rw; x++) {
        const global = (y + y0) * w + x + x0,
          local = y * rw + x;
        if (dist.data32F[local] < best[global]) {
          best[global] = dist.data32F[local];
          owner[global] = g;
        }
      }
      for (const m of [localArt, art, inv, dist]) {
        m.delete();
        mats.splice(mats.indexOf(m), 1);
      }
    }
    const output = [],
      issues = [];
    for (let g = 0; g < groups.length; g++) {
      progress(`Refining outline ${g + 1} / ${groups.length}`);
      const startMats = mats.length,
        group = groups[g],
        art = artFor(group),
        { x0, y0, rw, rh } = regions[g],
        localArt = own(art.roi(new cv.Rect(x0, y0, rw, rh))),
        shape = dilate(localArt, Math.round(p.border * mm));
      if (group.seedPaths?.length)
        fill(
          group.seedPaths.map((path) => path.map(([x, y]) => [x - x0, y - y0])),
          shape,
        );
      morph(shape, cv.MORPH_CLOSE, Math.round(p.bridge * mm));
      morph(shape, cv.MORPH_OPEN, Math.round(p.smooth * mm));
      if (p.smooth > 0) {
        cv.GaussianBlur(
          shape,
          shape,
          new cv.Size(0, 0),
          Math.max(0.1, (p.smooth * mm) / 3),
        );
        cv.threshold(shape, shape, 127, 255, cv.THRESH_BINARY);
      }
      cv.bitwise_or(
        shape,
        dilate(
          localArt,
          Math.max(1, Math.round(Math.min(p.border, 0.25) * mm)),
        ),
        shape,
      );
      const allowed = empty(rw, rh);
      for (let y = 0; y < rh; y++)
        for (let x = 0; x < rw; x++)
          allowed.data[y * rw + x] =
            owner[(y + y0) * w + x + x0] === g ? 255 : 0;
      const gapRadius = Math.round((p.gap * mm) / 2);
      if (gapRadius) cv.erode(allowed, allowed, kernel(gapRadius));
      cv.bitwise_and(shape, allowed, shape);
      const extract = (epsilon) =>
        contours(shape, epsilon).map((path) =>
          path.map(([x, y]) => [x + x0, y + y0]),
        );
      let paths = group.override || extract(p.detail * mm),
        solid = empty();
      fill(paths, solid);
      const nearSplitLine = (pixel, component) => {
        if (!component.splitLines?.length) return false;
        const x = pixel % w, y = (pixel - x) / w;
        return component.splitLines.some(({ points: [[ax, ay], [bx, by]], radius }) => {
          const dx = bx - ax, dy = by - ay,
            t = Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy)));
          return (x - ax - t * dx) ** 2 + (y - ay - t * dy) ** 2 <= radius * radius;
        });
      };
      const missing = () => {
        let n = 0;
        for (const id of group.componentIds) {
          const component = lookup.get(id);
          for (const [s, len] of component.runs)
            for (let i = s; i < s + len; i++)
              if (!solid.data[i] && !nearSplitLine(i, component)) n++;
        }
        return n;
      };
      let clipped = missing();
      if (clipped && !group.override) {
        paths = extract(0);
        solid.setTo(new cv.Scalar(0));
        fill(paths, solid);
        clipped = missing();
      }
      if (clipped)
        issues.push(
          `Group ${g + 1}: ${clipped} detected artwork pixels outside its cut. Reduce spacing, merge groups, or edit the outline.`,
        );
      if (!paths.length) issues.push(`Group ${g + 1}: no usable outline.`);
      output.push({
        ...group,
        key: groupKey(group.componentIds),
        paths,
        clipped,
      });
      while (mats.length > startMats) mats.pop().delete();
    }
    return {
      width: w,
      height: h,
      pxPerMM: mm,
      mode,
      params: p,
      components,
      groups: output,
      issues,
    };
  } finally {
    for (const m of mats.reverse()) m.delete();
  }
}
