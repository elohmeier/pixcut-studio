import { Canvas, Circle, FabricImage, Polygon, controlsUtils } from "fabric";
import {
  DEFAULTS,
  mergeGroups,
  splitGroup,
  moveComponents,
} from "./autocut.js";
import { runAnalysis, cancelAnalysis } from "./autocut-client.js";
import { analysisRaster } from "./analysis-raster.js";
import {
  validateLayout,
  validateSpacing,
  transformPoint,
  PX_PER_MM,
} from "./geometry.js";

export const cutPaths = (data) =>
  data.analysis.groups
    .flatMap((g) => g.paths)
    .map((path) =>
      path.map(([x, y]) => [
        (x - data.pad) / data.factor - data.sourceWidth / 2,
        (y - data.pad) / data.factor - data.sourceHeight / 2,
      ]),
    );
export function installCutEditor({ selected, changed, notice }) {
  const $ = (id) => document.getElementById(id),
    dialog = $("cut-dialog");
  let view,
    object,
    draft,
    imageURL,
    working = false,
    opening = false,
    editing = false,
    splitting = null,
    history = [],
    future = [],
    chosen = [0],
    pathIndex = 0;
  const selectIds = (id) =>
    [...$(id).selectedOptions].map((o) => Number(o.value));
  const message = (text) => ($("cut-message").textContent = text);
  const readParams = () =>
    Object.fromEntries(
      Object.keys(DEFAULTS).map((key) => {
        const el = $(`ac-${key}`);
        return [
          key,
          el.type === "checkbox"
            ? el.checked
            : el.type === "number"
              ? Number(el.value)
              : el.value,
        ];
      }),
    );
  const writeParams = (p) => {
    for (const [key, value] of Object.entries(p)) {
      const el = $(`ac-${key}`);
      if (!el) continue;
      if (el.type === "checkbox") el.checked = value;
      else el.value = value;
    }
  };
  const snapshot = () => {
    const { rgba, ...rest } = draft;
    return { ...structuredClone(rest), rgba };
  };
  function checkpoint() {
    history.push(snapshot());
    if (history.length > 20) history.shift();
    future = [];
  }
  function state() {
    for (const el of dialog.querySelectorAll("button,input,select"))
      el.disabled = working;
    for (const id of [
      "ac-merge",
      "ac-split",
      "ac-split-touching",
      "ac-assign",
      "ac-exclude",
      "ac-restore",
      "ac-edit",
      "ac-reset",
      "ac-insert",
      "ac-remove",
    ])
      $(id).disabled ||= !draft?.analysis;
    $("ac-cancel").disabled = !working;
    $("ac-undo").disabled = working || !history.length;
    $("ac-redo").disabled = working || !future.length;
    $("ac-apply").disabled = working || !draft?.analysis;
    $("ac-edit").textContent = editing
      ? "Stop editing points"
      : "Edit outline points";
    $("ac-split-touching").textContent = splitting
      ? "Cancel separator line"
      : "Split touching stickers with a line";
    if (splitting) for (const el of dialog.querySelectorAll("button,input,select"))
      if (!['ac-split-touching', 'ac-close'].includes(el.id)) el.disabled = true;
  }
  function options(id, items, selection = []) {
    $(id).replaceChildren(
      ...items.map(([value, label]) => {
        const option = new Option(label, String(value));
        option.selected = selection.includes(value);
        return option;
      }),
    );
  }
  function render() {
    if (!draft?.analysis) return;
    const { groups, components, issues } = draft.analysis;
    chosen = chosen.filter((i) => groups[i]);
    if (!chosen.length && groups.length) chosen = [0];
    options(
      "ac-groups",
      groups.map((g, i) => [
        i,
        `${i + 1} · ${g.componentIds.length} pieces · ${g.paths.length} outline${g.paths.length === 1 ? "" : "s"}${g.override ? " · locked" : ""}${g.clipped ? " · CHECK" : ""}`,
      ]),
      chosen,
    );
    const g = groups[chosen[0]];
    options(
      "ac-components",
      (g?.componentIds || []).map((id) => {
        const c = components.find((c) => c.id === id);
        return [
          id,
          `Piece ${id} · ${(c.area / draft.analysis.pxPerMM ** 2).toFixed(2)} mm²`,
        ];
      }),
    );
    options(
      "ac-destination",
      [
        [-1, "New separate group"],
        ...groups.map((_, i) => [i, `Group ${i + 1}`]),
      ],
      [chosen[0]],
    );
    pathIndex = Math.min(pathIndex, Math.max(0, (g?.paths.length || 1) - 1));
    options(
      "ac-path",
      (g?.paths || []).map((_, i) => [i, `Outline ${i + 1}`]),
      [pathIndex],
    );
    options(
      "ac-point",
      (g?.paths[pathIndex] || []).map((_, i) => [i, `Point ${i + 1}`]),
      [0],
    );
    for (const o of view.getObjects().slice(1)) view.remove(o);
    groups.forEach((group, i) =>
      group.paths.forEach((path, j) => {
        const active = chosen.includes(i),
          editable = editing && chosen[0] === i && pathIndex === j;
        const poly = new Polygon(
          path.map(([x, y]) => ({ x, y })),
          {
            fill: active ? "rgba(0,137,96,.09)" : "transparent",
            stroke: active ? "#ba5625" : "#008960",
            strokeWidth: editable ? 1 : 2,
            strokeUniform: true,
            objectCaching: false,
            selectable: editable,
            evented: editable,
            hasBorders: false,
            lockMovementX: true,
            lockMovementY: true,
            cornerSize: 14,
            cornerStyle: "circle",
            cornerColor: "#ba5625",
            transparentCorners: false,
          },
        );
        // Polygon points are absolute raster positions; Fabric computes its center.
        if (editable) {
          poly.controls = controlsUtils.createPolyControls(poly);
          poly.on("modified", () => {
            const next = poly.points.map((p) =>
              transformPoint(
                [p.x - poly.pathOffset.x, p.y - poly.pathOffset.y],
                poly.calcTransformMatrix(),
              ),
            );
            checkpoint();
            group.paths[j] = next;
            group.override = structuredClone(group.paths);
            group.clipped = 0;
            message(
              "Manual outline locked. Apply validates it against detected artwork.",
            );
            render();
          });
        }
        view.add(poly);
        if (editable) view.setActiveObject(poly);
      }),
    );
    for (const [x, y] of splitting?.points || []) view.add(new Circle({
      left: x, top: y, radius: 5, originX: "center", originY: "center",
      fill: "#ba5625", stroke: "white", strokeWidth: 2,
      selectable: false, evented: false,
    }));
    view.renderAll();
    state();
    const included = new Set(groups.flatMap((g) => g.componentIds)),
      excluded = components.filter((c) => !included.has(c.id)).length;
    $("ac-summary").textContent =
      `${components.length} artwork pieces → ${groups.length} groups / ${groups.reduce((n, g) => n + g.paths.length, 0)} outlines · ${draft.analysis.mode} detection${excluded ? ` · ${excluded} excluded (still printed)` : ""}`;
    $("ac-issues").textContent = issues?.length
      ? issues.join("\n")
      : "No detected artwork is clipped. Review touching stickers and fine details before applying.";
  }
  async function process({
    detect = false,
    seeds = false,
    groups,
    splitTouching,
    save = true,
  } = {}) {
    if (working) return;
    const params = readParams();
    working = true;
    state();
    message("Loading local image analysis…");
    $("ac-summary").textContent = "Loading local image analysis…";
    try {
      const request = {
        width: draft.width,
        height: draft.height,
        pxPerMM: draft.pxPerMM,
        params,
      };
      if (detect || !draft.analysis) {
        request.rgba = draft.rgba;
        request.contentRect = [
          draft.pad,
          draft.pad,
          Math.round(draft.sourceWidth * draft.factor),
          Math.round(draft.sourceHeight * draft.factor),
        ];
        if (seeds) request.seedPaths = draft.seeds;
      } else {
        request.components = draft.analysis.components;
        request.mode = draft.analysis.mode;
        request.groups = groups || draft.analysis.groups;
        if (splitTouching) request.splitTouching = splitTouching;
      }
      const result = await runAnalysis(request, (step) => {
        message(step);
        $("ac-summary").textContent = `${step}…`;
      });
      if (save) checkpoint();
      draft.analysis = result;
      writeParams(result.params);
      editing = false;
      splitting = null;
      message(
        "Contours updated. Click the sheet to select a group; Shift-click selects several.",
      );
      return true;
    } catch (e) {
      message(e.message);
      if (!draft.analysis) $("ac-summary").textContent = "Detection stopped";
      return false;
    } finally {
      working = false;
      render();
      state();
    }
  }
  async function open() {
    if (opening || dialog.open) return;
    opening = true;
    try {
      object = selected();
      if (
        Math.abs(object.scaleX - object.scaleY) > 1e-5 ||
        object.skewX ||
        object.skewY
      )
        throw Error(
          "Fit the artwork to page first: automatic cuts currently require uniform scaling without skew.",
        );
      history = [];
      future = [];
      chosen = [0];
      pathIndex = 0;
      editing = false;
      splitting = null;
      const rasterSpec = analysisRaster(object.width, object.height, object.scaleX),
        { factor, pad, pxPerMM: mm } = rasterSpec;
      const raster = document.createElement("canvas");
      raster.width = rasterSpec.width;
      raster.height = rasterSpec.height;
      const ctx = raster.getContext("2d", { willReadFrequently: true });
      ctx.drawImage(
        object.getElement(),
        pad,
        pad,
        rasterSpec.contentWidth,
        rasterSpec.contentHeight,
      );
      draft = object.autoCut
        ? structuredClone(object.autoCut)
        : {
            factor,
            pad,
            width: raster.width,
            height: raster.height,
            pxPerMM: mm,
            sourceWidth: object.width,
            sourceHeight: object.height,
            scaleAtAnalysis: object.scaleX,
          };
      // Retain the original analysis grid if artwork was subsequently resized.
      if (object.autoCut) {
        raster.width = draft.width;
        raster.height = draft.height;
        ctx.drawImage(
          object.getElement(),
          draft.pad,
          draft.pad,
          Math.round(object.width * draft.factor),
          Math.round(object.height * draft.factor),
        );
        draft.pxPerMM = (PX_PER_MM * draft.factor) / object.scaleX;
        draft.scaleAtAnalysis = object.scaleX;
      }
      draft.rgba = ctx.getImageData(0, 0, raster.width, raster.height).data;
      draft.seeds = object.customPaths?.map((path) =>
        path.map(([x, y]) => [
          (x + object.width / 2) * draft.factor + draft.pad,
          (y + object.height / 2) * draft.factor + draft.pad,
        ]),
      );
      if (imageURL) URL.revokeObjectURL(imageURL);
      imageURL = URL.createObjectURL(
        await new Promise((resolve) => raster.toBlob(resolve)),
      );
      if (view) {
        view.cancelRequestedRender();
        await view.dispose();
      }
      view = new Canvas("cut-canvas", {
        width: draft.width,
        height: draft.height,
        selection: false,
        enableRetinaScaling: false,
      });
      const bg = await FabricImage.fromURL(imageURL);
      bg.set({
        left: 0,
        top: 0,
        originX: "left",
        originY: "top",
        selectable: false,
        evented: false,
      });
      view.add(bg);
      $("ac-stage").style.setProperty(
        "--cut-ratio",
        `${draft.width}/${draft.height}`,
      );
      view.on("mouse:down", (event) => {
        if (working || editing || !draft.analysis) return;
        const x = Math.round(event.scenePoint.x),
          y = Math.round(event.scenePoint.y),
          at = y * draft.width + x;
        if (x < 0 || x >= draft.width || y < 0 || y >= draft.height) return;
        if (splitting) {
          splitting.points.push([x, y]);
          if (splitting.points.length === 1) {
            message("First end marked. Click the other end of the shared white border.");
            render();
          } else {
            const splitTouching = splitting;
            splitting = null;
            process({ splitTouching });
          }
          return;
        }
        const component = draft.analysis.components.find((c) =>
          c.runs.some(([s, n]) => at >= s && at < s + n),
        );
        let index = component
          ? draft.analysis.groups.findIndex((g) =>
              g.componentIds.includes(component.id),
            )
          : -1;
        if (index < 0)
          index = draft.analysis.groups.findIndex((g) =>
            g.paths.some((path) => inside([x, y], path)),
          );
        if (index < 0) return;
        chosen = event.e.shiftKey
          ? chosen.includes(index)
            ? chosen.filter((i) => i !== index)
            : [...chosen, index]
          : [index];
        render();
        if (component)
          for (const option of $("ac-components").options)
            option.selected = Number(option.value) === component.id;
      });
      writeParams(draft.analysis?.params || DEFAULTS);
      $("ac-seeds").hidden = !draft.seeds?.length;
      dialog.showModal();
      state();
      if (draft.analysis) {
        render();
        message(
          "Existing outlines loaded. Update contours after changing scale or parameters.",
        );
      } else
        await process({
          detect: true,
          seeds: !!draft.seeds?.length,
          save: false,
        });
    } catch (e) {
      notice(e.message || String(e));
    } finally {
      opening = false;
    }
  }
  function inside([x, y], path) {
    let hit = false;
    for (let i = 0, j = path.length - 1; i < path.length; j = i++) {
      const [xi, yi] = path[i],
        [xj, yj] = path[j];
      if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi)
        hit = !hit;
    }
    return hit;
  }
  $("auto-cut").onclick = open;
  $("ac-cancel").onclick = cancelAnalysis;
  $("ac-close").onclick = () => dialog.close();
  dialog.addEventListener("cancel", (e) => {
    if (working) {
      e.preventDefault();
      cancelAnalysis();
    }
  });
  $("ac-detect").onclick = () => {
    if (
      draft.analysis &&
      !confirm(
        "Detect again replaces grouping and manual outlines in this workspace. You can undo it. Continue?",
      )
    )
      return;
    process({ detect: true });
  };
  $("ac-seeds").onclick = () => {
    if (
      draft.analysis &&
      !confirm(
        "Use existing custom contours as grouping regions? This replaces the draft grouping; you can undo it.",
      )
    )
      return;
    process({ detect: true, seeds: true });
  };
  $("ac-update").onclick = () => process();
  $("ac-preset").onchange = () => {
    const p = readParams(),
      preset = $("ac-preset").value;
    writeParams({
      ...p,
      join: preset === "rounded" ? 1.5 : 1.1,
      keepSmall: preset === "separate",
      bridge: preset === "rounded" ? 1.3 : 1.1,
    });
    message(
      "Preset selected. Click Detect again to rebuild grouping; existing edits remain until then.",
    );
  };
  $("ac-groups").onchange = () => {
    chosen = selectIds("ac-groups");
    editing = false;
    splitting = null;
    render();
  };
  $("ac-path").onchange = () => {
    pathIndex = Number($("ac-path").value);
    render();
  };
  const action = (fn) => {
    try {
      fn();
    } catch (e) {
      message(e.message);
    }
  };
  $("ac-merge").onclick = () =>
    action(() =>
      process({ groups: mergeGroups(draft.analysis.groups, chosen) }),
    );
  $("ac-split").onclick = () =>
    action(() =>
      process({ groups: splitGroup(draft.analysis.groups, chosen[0]) }),
    );
  $("ac-split-touching").onclick = () => action(() => {
    if (splitting) {
      splitting = null;
      message("Separator line cancelled.");
      render();
      return;
    }
    if (chosen.length !== 1) throw Error("Select one group to split.");
    const group = draft.analysis.groups[chosen[0]];
    if (group.override) throw Error("Reset the locked outline before splitting its group.");
    const selectedComponent = selectIds("ac-components")[0],
      componentId = selectedComponent ?? group.componentIds
        .map((id) => draft.analysis.components.find((c) => c.id === id))
        .sort((a, b) => b.area - a.area)[0].id;
    editing = false;
    splitting = { componentId, points: [] };
    message("Click the two ends of the shared white border between the stickers.");
    render();
  });
  $("ac-assign").onclick = () =>
    action(() =>
      process({
        groups: moveComponents(
          draft.analysis.groups,
          selectIds("ac-components"),
          Number($("ac-destination").value),
        ),
      }),
    );
  $("ac-exclude").onclick = () =>
    action(() => {
      const ids = new Set(selectIds("ac-components"));
      if (!ids.size) throw Error("Select artwork pieces to exclude.");
      if (
        !confirm(
          "Exclude these pieces from cutting? Their artwork will still be printed.",
        )
      )
        return;
      const groups = draft.analysis.groups;
      if (
        groups.some(
          (g) => g.override && g.componentIds.some((id) => ids.has(id)),
        )
      )
        throw Error("Reset affected locked outlines first.");
      process({
        groups: groups
          .map((g) => ({
            ...g,
            componentIds: g.componentIds.filter((id) => !ids.has(id)),
            seedPaths: undefined,
          }))
          .filter((g) => g.componentIds.length),
      });
    });
  $("ac-restore").onclick = () => {
    const groups = draft.analysis.groups,
      used = new Set(groups.flatMap((g) => g.componentIds));
    process({
      groups: [
        ...groups,
        ...draft.analysis.components
          .filter((c) => !used.has(c.id))
          .map((c) => ({ componentIds: [c.id] })),
      ],
    });
  };
  $("ac-edit").onclick = () => {
    editing = !editing;
    render();
    message(
      "Drag a vertex. Edits lock this group against regeneration. Use the point selector to insert or delete a vertex.",
    );
  };
  $("ac-reset").onclick = () => {
    if (!confirm("Reset the selected groups to generated outlines?")) return;
    const groups = structuredClone(draft.analysis.groups);
    for (const i of chosen) delete groups[i].override;
    process({ groups });
  };
  function editPoint(remove) {
    const g = draft.analysis.groups[chosen[0]],
      path = g?.paths[pathIndex];
    if (!path) return;
    const i = Number($("ac-point").value);
    if (remove && path.length <= 3)
      return message("A closed outline needs at least three vertices.");
    checkpoint();
    if (remove) path.splice(i, 1);
    else {
      const a = path[i],
        b = path[(i + 1) % path.length];
      path.splice(i + 1, 0, [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]);
    }
    g.override = structuredClone(g.paths);
    editing = true;
    render();
  }
  $("ac-insert").onclick = () => editPoint(false);
  $("ac-remove").onclick = () => editPoint(true);
  $("ac-undo").onclick = () => {
    if (!history.length) return;
    future.push(snapshot());
    draft = history.pop();
    writeParams(draft.analysis?.params || DEFAULTS);
    editing = false;
    splitting = null;
    render();
  };
  $("ac-redo").onclick = () => {
    if (!future.length) return;
    history.push(snapshot());
    draft = future.pop();
    writeParams(draft.analysis?.params || DEFAULTS);
    editing = false;
    splitting = null;
    render();
  };
  $("ac-apply").onclick = async () => {
    if (!(await process({ save: false }))) return;
    try {
      if (draft.analysis.issues.length)
        throw Error(
          "Resolve the highlighted artwork conflicts before applying.",
        );
      const local = cutPaths(draft),
        scene = local.map((path) =>
          path.map((p) => transformPoint(p, object.calcTransformMatrix())),
        );
      validateLayout(scene);
      validateSpacing(scene, draft.analysis.params.gap * PX_PER_MM);
      const saved = structuredClone(draft);
      delete saved.rgba;
      delete saved.seeds;
      object.autoCut = saved;
      object.cutMode = "auto";
      object.borderMM = 0;
      changed();
      dialog.close();
      notice(
        `Applied ${scene.length} automatic cut outlines. Review the sheet before printing.`,
      );
    } catch (e) {
      message(e.message);
    }
  };
}
