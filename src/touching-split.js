// Divide one connected component at a short line drawn through a shared
// sticker rim. The removed narrow seam is still printed, but is not treated
// as artwork that must belong to either cut.
export function splitTouchingComponent(cv, width, height, components, groups, componentId, points, radius) {
  const component = components.find((c) => c.id === componentId),
    groupIndex = groups.findIndex((g) => g.componentIds.includes(componentId));
  if (!component || groupIndex < 0 || groups[groupIndex].override ||
      !Array.isArray(points) || points.length !== 2 ||
      points.some((p) => !Array.isArray(p) || p.length !== 2 || !p.every(Number.isFinite)) ||
      !Number.isInteger(radius) || radius < 1 || radius > 200 ||
      (component?.splitLines?.length || 0) >= 20)
    throw Error("Select an unlocked artwork piece and two ends of its shared border.");
  const [left, top, rw, rh] = component.bounds,
    mask = new Uint8Array(rw * rh);
  for (const [start, length] of component.runs) for (let at = start; at < start + length; at++) {
    const x = at % width, y = (at - x) / width;
    mask[(y - top) * rw + x - left] = 255;
  }
  const [ax, ay] = points[0].map((v, i) => v - (i ? top : left)),
    [bx, by] = points[1].map((v, i) => v - (i ? top : left)),
    dx = bx - ax, dy = by - ay, length2 = dx * dx + dy * dy;
  if (length2 < 100) throw Error("Draw a line across the shared border, at least 10 pixels long.");
  const x0 = Math.max(0, Math.floor(Math.min(ax, bx) - radius)),
    y0 = Math.max(0, Math.floor(Math.min(ay, by) - radius)),
    x1 = Math.min(rw - 1, Math.ceil(Math.max(ax, bx) + radius)),
    y1 = Math.min(rh - 1, Math.ceil(Math.max(ay, by) + radius));
  for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
    const t = Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / length2)),
      distance2 = (x - ax - t * dx) ** 2 + (y - ay - t * dy) ** 2;
    if (distance2 <= radius * radius) mask[y * rw + x] = 0;
  }
  const source = cv.matFromArray(rh, rw, cv.CV_8UC1, mask),
    labels = new cv.Mat(), stats = new cv.Mat(), centers = new cv.Mat();
  try {
    const count = cv.connectedComponentsWithStats(source, labels, stats, centers, 8, cv.CV_32S),
      ranked = [];
    for (let id = 1; id < count; id++) ranked.push({ id, area: stats.data32S[id * 5 + 4] });
    ranked.sort((a, b) => b.area - a.area);
    if (ranked.length < 2 || ranked[1].area < Math.max(50, component.area * 0.01))
      throw Error("The line did not separate two stickers. Extend it across the shared border.");
    const labelOwner = new Uint8Array(count);
    labelOwner[ranked[0].id] = 1; labelOwner[ranked[1].id] = 2;
    for (const { id } of ranked.slice(2)) {
      const x = centers.data64F[id * 2], y = centers.data64F[id * 2 + 1],
        first = (x - centers.data64F[ranked[0].id * 2]) ** 2 +
          (y - centers.data64F[ranked[0].id * 2 + 1]) ** 2,
        second = (x - centers.data64F[ranked[1].id * 2]) ** 2 +
          (y - centers.data64F[ranked[1].id * 2 + 1]) ** 2;
      labelOwner[id] = first <= second ? 1 : 2;
    }
    const firstId = Math.max(...components.map((c) => c.id)) + 1,
      replacements = [firstId, firstId + 1].map((id) => ({
        id, area: 0, bounds: [width, height, 0, 0], center: [0, 0], runs: [],
        splitLines: [...(component.splitLines || []), { points, radius: radius + 3 }],
      }));
    for (let y = 0; y < rh; y++) for (let x = 0; x < rw;) {
      const owner = labelOwner[labels.data32S[y * rw + x]];
      if (!owner) { x++; continue; }
      const start = x, c = replacements[owner - 1];
      while (x < rw && labelOwner[labels.data32S[y * rw + x]] === owner) {
        c.area++; c.center[0] += left + x; c.center[1] += top + y;
        c.bounds[0] = Math.min(c.bounds[0], left + x);
        c.bounds[1] = Math.min(c.bounds[1], top + y);
        c.bounds[2] = Math.max(c.bounds[2], left + x);
        c.bounds[3] = Math.max(c.bounds[3], top + y);
        x++;
      }
      c.runs.push([(top + y) * width + left + start, x - start]);
    }
    for (const c of replacements) {
      c.center = c.center.map((v) => v / c.area);
      c.bounds[2] -= c.bounds[0] - 1;
      c.bounds[3] -= c.bounds[1] - 1;
    }
    const sourceGroup = groups[groupIndex],
      firstArea = replacements[0].area, secondArea = replacements[1].area,
      firstKeepsGroup = firstArea >= secondArea,
      keptId = firstKeepsGroup ? firstId : firstId + 1,
      newId = firstKeepsGroup ? firstId + 1 : firstId;
    const nextGroups = groups.map((g, i) => i === groupIndex ? {
      ...sourceGroup,
      componentIds: sourceGroup.componentIds.map((id) => id === componentId ? keptId : id),
      seedPaths: undefined,
    } : g);
    nextGroups.splice(groupIndex + 1, 0, { componentIds: [newId] });
    return {
      components: [...components.filter((c) => c.id !== componentId), ...replacements],
      groups: nextGroups,
    };
  } finally {
    source.delete(); labels.delete(); stats.delete(); centers.delete();
  }
}
