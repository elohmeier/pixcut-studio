import cvModule from "@techstark/opencv-js";
import { analyze } from "./autocut.js";
import { openCVReady } from "./opencv-ready.js";
const ready = openCVReady(cvModule);
self.onmessage = async ({ data }) => {
  try {
    const { cv } = await ready;
    const result = analyze(cv, data.request, (message) =>
      self.postMessage({ id: data.id, progress: message }),
    );
    self.postMessage({ id: data.id, result });
  } catch (e) {
    self.postMessage({ id: data.id, error: e.message || String(e) });
  }
};
