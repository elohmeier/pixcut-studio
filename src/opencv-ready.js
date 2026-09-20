// OpenCV 4.12 exposes a self-resolving Emscripten thenable, not a Promise.
// Resolve a wrapper to avoid Promise assimilation looping on Module.then.
export function openCVReady(module) {
  return new Promise((resolve, reject) => {
    const done = (cv) => resolve({ cv });
    if (module.Mat) done(module);
    else if (typeof module.then === "function") module.then(done, reject);
    else module.onRuntimeInitialized = () => done(module);
  });
}
