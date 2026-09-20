let worker,
  pending,
  sequence = 0;
export function cancelAnalysis() {
  worker?.terminate();
  worker = null;
  clearTimeout(pending?.timer);
  pending?.reject(Error("Analysis cancelled; existing cuts are unchanged."));
  pending = null;
}
export function runAnalysis(request, onProgress = () => {}) {
  if (pending) throw Error("Analysis is already running.");
  if (!worker) {
    worker = new Worker(new URL("./autocut.worker.js", import.meta.url), {
      type: "module",
    });
    worker.onmessage = ({ data }) => {
      if (data.id !== pending?.id) return;
      if (data.progress) {
        pending.onProgress(data.progress);
        return;
      }
      const p = pending;
      pending = null;
      clearTimeout(p.timer);
      data.error ? p.reject(Error(data.error)) : p.resolve(data.result);
    };
    worker.onerror = () => cancelAnalysis();
  }
  return new Promise((resolve, reject) => {
    pending = {
      id: ++sequence,
      resolve,
      reject,
      onProgress,
      timer: setTimeout(cancelAnalysis, 120000),
    };
    worker.postMessage({ id: sequence, request });
  });
}
