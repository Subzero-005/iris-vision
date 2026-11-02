import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

/* ───────────────────────── config ───────────────────────── */

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const OPENCV_URL = "https://docs.opencv.org/4.9.0/opencv.js";

const IRIS_DIAMETER_MM = 11.7;     // horizontal visible iris diameter, near-constant across adults
const ASSUMED_HFOV_DEG = 60;       // typical laptop webcam horizontal field of view
const EAR_CLOSE = 0.19, EAR_OPEN = 0.23, EAR_OCCLUDED = 0.17;
const PASS_THRESHOLD = 0.6;
const HISTORY = 180;
const ANALYSIS_W = 160, ANALYSIS_H = 90;   // eye ROI resolution used for validation

// MediaPipe Face Mesh indices. "right"/"left" are the subject's own eyes.
const EYES = {
  right: {
    center: 468, iris: [469, 470, 471, 472],
    outer: 33, inner: 133, top: 159, bottom: 145,
    ear: [[160, 144], [158, 153]],
    contour: [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
  },
  left: {
    center: 473, iris: [474, 475, 476, 477],
    outer: 263, inner: 362, top: 386, bottom: 374,
    ear: [[385, 380], [387, 373]],
    contour: [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466],
  },
};

const COLORS = { cyan: "#3fe0d0", violet: "#8b7bff", pink: "#ff5fa2", amber: "#ffb547", green: "#3ddc84", red: "#ff5b6b", muted: "#8a97ab", line: "#1c2534" };

/* ───────────────────────── dom ───────────────────────── */

const $ = (id) => document.getElementById(id);
const stage = $("stage"), video = $("video"), canvas = $("output");
const ctx = canvas.getContext("2d", { willReadFrequently: true });
const eyeRow = document.querySelector(".eye-row");
const crops = { right: $("crop-right"), left: $("crop-left") };
const cropCtx = { right: crops.right.getContext("2d"), left: crops.left.getContext("2d") };
const analysisCanvas = document.createElement("canvas");
analysisCanvas.width = ANALYSIS_W; analysisCanvas.height = ANALYSIS_H;
const aCtx = analysisCanvas.getContext("2d", { willReadFrequently: true });

/* ───────────────────────── state ───────────────────────── */

const state = {
  landmarker: null, runningMode: "VIDEO", cv: null,
  source: null, stream: null, raf: 0, lastVideoTime: -1,
  image: null, lastResult: null, lastAnalysis: null,
  toggles: { mesh: false, contours: true, iris: true, validators: true, gaze: true },
  fps: 0, lastFrameAt: 0, latency: 0, lastUiAt: 0, errorShown: false,
};
let session;

function resetSession() {
  session = {
    startedAt: performance.now(), frames: 0, blinks: 0, eyeClosed: false,
    pass: 0, judged: 0,
    ema: { right: null, left: null }, gaze: null, gazeOffset: session?.gazeOffset ?? { x: 0, y: 0 },
    hist: { earR: [], earL: [], scoreR: [], scoreL: [] }, gazeTrail: [],
  };
}
resetSession();

/* ───────────────────────── loading ───────────────────────── */

const modelReady = (async () => {
  setChip("chipModel", "loading", "Landmarker");
  const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
  const options = (delegate) => ({
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: "VIDEO", numFaces: 1,
    minFaceDetectionConfidence: 0.5, minTrackingConfidence: 0.5,
  });
  try {
    state.landmarker = await FaceLandmarker.createFromOptions(fileset, options("GPU"));
  } catch {
    state.landmarker = await FaceLandmarker.createFromOptions(fileset, options("CPU"));
  }
  setChip("chipModel", "ok", "Landmarker");
  $("btnCamera").disabled = false;
  $("stageMsg").textContent = "Models ready.";
})().catch((err) => {
  console.error(err);
  setChip("chipModel", "err", "Landmarker failed");
  $("stageMsg").textContent = "Could not load the face model. Check your connection and reload.";
  throw err;
});

function loadOpenCV() {
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = OPENCV_URL;
    script.async = true;
    script.onerror = () => reject(new Error("OpenCV.js failed to load"));
    script.onload = () => {
      let done = false;
      const finish = (cv) => { if (!done) { done = true; resolve({ cv }); } };
      const cv = window.cv;
      if (!cv) return reject(new Error("OpenCV.js global missing"));
      if (cv.Mat) return finish(cv);
      // Emscripten MODULARIZE builds expose a thenable Module; wrap to avoid recursive thenable resolution.
      if (typeof cv.then === "function") cv.then((m) => finish(m));
      cv.onRuntimeInitialized = () => finish(window.cv.Mat ? window.cv : cv);
    };
    document.head.appendChild(script);
  });
}

setChip("chipCv", "loading", "OpenCV.js");
loadOpenCV()
  .then(({ cv }) => {
    if (typeof cv.HoughCircles !== "function") throw new Error("HoughCircles not in this build");
    state.cv = cv;
    setChip("chipCv", "ok", "OpenCV.js");
    rerenderImage();
  })
  .catch((err) => {
    console.warn(err);
    setChip("chipCv", "err", "OpenCV off · Daugman only");
  });

async function setRunningMode(mode) {
  if (state.runningMode === mode) return;
  await state.landmarker.setOptions({ runningMode: mode });
  state.runningMode = mode;
}

/* ───────────────────────── sources ───────────────────────── */

async function startCamera() {
  const btn = $("btnCamera");
  btn.disabled = true;
  try {
    await modelReady;
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("Camera API unavailable (requires HTTPS).");
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }, audio: false,
    });
    state.stream = stream;
    video.srcObject = stream;
    await video.play();
    await setRunningMode("VIDEO");
    state.source = "camera";
    state.image = null;
    state.lastVideoTime = -1;
    resetSession();
    setMirrored(true);
    stage.classList.add("active");
    setActionsEnabled(true);
    btn.querySelector("span").textContent = "Stop camera";
    btn.classList.replace("btn-primary", "btn-danger");
    state.raf = requestAnimationFrame(loop);
  } catch (err) {
    console.error(err);
    const msg = err.name === "NotAllowedError" ? "Camera permission was denied." :
      err.name === "NotFoundError" ? "No camera found — try uploading an image instead." : err.message;
    toast(msg, true);
  } finally {
    btn.disabled = false;
  }
}

function stopCamera() {
  cancelAnimationFrame(state.raf);
  state.stream?.getTracks().forEach((t) => t.stop());
  state.stream = null;
  video.srcObject = null;
  if (state.source === "camera") state.source = "paused";
  const btn = $("btnCamera");
  btn.querySelector("span").textContent = "Start camera";
  btn.classList.replace("btn-danger", "btn-primary");
}

function loop() {
  if (state.source !== "camera") return;
  if (video.readyState >= 2 && video.currentTime !== state.lastVideoTime) {
    state.lastVideoTime = video.currentTime;
    const t0 = performance.now();
    try {
      const result = state.landmarker.detectForVideo(video, t0);
      processFrame(video, video.videoWidth, video.videoHeight, result, true);
    } catch (err) {
      console.error(err);
      if (!state.errorShown) { toast("Frame processing error — see console.", true); state.errorShown = true; }
    }
    state.latency = ema(state.latency, performance.now() - t0, 0.1);
  }
  state.raf = requestAnimationFrame(loop);
}

function analyzeImageFile(file) {
  if (!file || !file.type.startsWith("image/")) return toast("Please choose an image file.", true);
  return analyzeImageUrl(URL.createObjectURL(file));
}

async function analyzeImageUrl(url) {
  stopCamera();
  const img = new Image();
  img.src = url;
  try {
    await img.decode();
    await modelReady;
    await setRunningMode("IMAGE");
    const t0 = performance.now();
    const result = state.landmarker.detect(img);
    state.latency = performance.now() - t0;
    state.source = "image";
    state.image = img;
    state.lastResult = result;
    resetSession();
    setMirrored(false);
    stage.classList.add("active");
    setActionsEnabled(true);
    processFrame(img, img.naturalWidth, img.naturalHeight, result, false);
    if (!result.faceLandmarks?.length) toast("No face found in that image.", true);
  } catch (err) {
    console.error(err);
    toast("Could not analyze that image.", true);
  }
}

function rerenderImage() {
  if (state.source === "image" && state.image && state.lastResult) {
    processFrame(state.image, state.image.naturalWidth, state.image.naturalHeight, state.lastResult, false);
  }
}

/* ───────────────────────── frame processing ───────────────────────── */

function processFrame(src, w, h, result, live) {
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  ctx.drawImage(src, 0, 0, w, h);

  const now = performance.now();
  if (live) {
    if (state.lastFrameAt) state.fps = ema(state.fps, 1000 / Math.max(1, now - state.lastFrameAt), 0.1);
    state.lastFrameAt = now;
    session.frames++;
  } else {
    // Still images are re-rendered on toggle changes; start their counters fresh each time.
    Object.assign(session, { frames: 1, pass: 0, judged: 0, gazeTrail: [], hist: { earR: [], earL: [], scoreR: [], scoreL: [] } });
  }

  const face = result.faceLandmarks?.[0];
  if (!face) {
    state.lastAnalysis = null;
    for (const side of ["right", "left"]) drawEmptyCrop(side);
    updateUi(null, live, true);
    return;
  }

  const pts = face.map((p) => ({ x: p.x * w, y: p.y * h, z: p.z * w }));
  const eyes = {};
  for (const side of ["right", "left"]) eyes[side] = analyzeEye(side, pts, live);

  const a = deriveFaceMetrics(eyes, w, live);
  state.lastAnalysis = { pts, eyes, ...a, w, h };

  drawOverlay(pts, eyes, w);
  for (const side of ["right", "left"]) drawCropOverlay(side, eyes[side], pts);
  updateUi(state.lastAnalysis, live, false);
}

function analyzeEye(side, pts, live) {
  const def = EYES[side];
  const c0 = pts[def.center];
  const boundary = def.iris.map((i) => pts[i]);
  const meanR = boundary.reduce((s, p) => s + dist(p, c0), 0) / boundary.length;

  let fit = fitCircle(boundary);
  if (!fit || !Number.isFinite(fit.r) || fit.r > meanR * 1.6 || fit.r < meanR * 0.6) {
    fit = { x: c0.x, y: c0.y, r: meanR, rms: NaN };
  }

  const outer = pts[def.outer], inner = pts[def.inner], top = pts[def.top], bottom = pts[def.bottom];
  const eyeW = dist(outer, inner);
  const ear = def.ear.reduce((s, [a, b]) => s + dist(pts[a], pts[b]), 0) / (2 * eyeW);

  // Gaze: iris position along the canthus axis and between the lids. +x = subject's right, +y = down.
  const t = project(fit, outer, inner);
  const v = project(fit, top, bottom);
  const gx = side === "right" ? (0.5 - t) * 2 : (t - 0.5) * 2;
  const gy = (v - 0.5) * 2;

  // 16:9 region of interest centred on the eye.
  const cx = (outer.x + inner.x) / 2, cy = (outer.y + inner.y + top.y + bottom.y) / 4;
  const roiW = eyeW * 1.7, roiH = roiW * 9 / 16;
  const roi = { x: cx - roiW / 2, y: cy - roiH / 2, w: roiW, h: roiH };

  // Raw crop for display (taken before overlays are drawn on the main canvas).
  drawRoi(cropCtx[side], canvas, roi, crops[side].width, crops[side].height);

  const validation = validateEye(side, fit, roi, def.contour.map((i) => pts[i]), ear, live);
  return { side, fit, ear, gx, gy, eyeW, roi, ...validation };
}

function validateEye(side, fit, roi, contour, ear, live) {
  drawRoi(aCtx, canvas, roi, ANALYSIS_W, ANALYSIS_H);
  const img = aCtx.getImageData(0, 0, ANALYSIS_W, ANALYSIS_H);
  const s = ANALYSIS_W / roi.w;
  const cs = { x: (fit.x - roi.x) * s, y: (fit.y - roi.y) * s };
  const rs = fit.r * s;
  const poly = contour.map((p) => ({ x: (p.x - roi.x) * s, y: (p.y - roi.y) * s }));

  const contrast = radialContrast(img, cs, rs, poly);
  const daugman = daugmanCheck(img, cs, rs);
  const hough = houghCheck(img, cs, rs);
  const toFrame = (m) => (m && !m.none ? { x: m.x / s + roi.x, y: m.y / s + roi.y, r: m.r / s, err: m.err, ratio: m.ratio } : m);

  const openness = clamp((ear - 0.15) / 0.12);
  const kScore = contrast == null ? 0 : clamp(contrast / 0.25);
  const gD = agreement(daugman), gH = agreement(hough);
  const geom = gH == null ? gD : 0.65 * gD + 0.35 * gH;

  let score = 0.45 * geom + 0.35 * kScore + 0.2 * openness;
  if (live) {
    session.ema[side] = session.ema[side] == null ? score : ema(session.ema[side], score, 0.3);
    score = session.ema[side];
  }

  let verdict, reason;
  if (ear < EAR_OCCLUDED) { verdict = "occl"; reason = "eyelid closed"; }
  else if (score >= PASS_THRESHOLD) { verdict = "pass"; reason = "fit confirmed"; }
  else {
    verdict = "fail";
    const parts = [["low contrast", kScore], ["partly closed", openness], [daugman.none ? "no limbus edge" : "circle mismatch", geom]];
    reason = parts.sort((a, b) => a[1] - b[1])[0][0];
  }
  return { contrast, daugman: toFrame(daugman), hough: toFrame(hough), geom, kScore, openness, score, verdict, reason };
}

// 1 when a detector agrees with the landmark fit, 0 when it disagrees or finds nothing; null if unavailable.
function agreement(m) {
  if (!m) return null;
  if (m.none) return 0;
  return clamp(1 - m.err / 0.6) * clamp(1 - Math.abs(1 - m.ratio) / 0.5);
}

// Daugman's integro-differential operator, restricted to the lateral limbus arcs that eyelids rarely cover:
// max over (x, y, r) of the radial derivative of the mean intensity along the circle.
const DAUGMAN_ANGLES = [];
for (let i = 0; i < 12; i++) {
  const t = (-25 + i * 6) * Math.PI / 180;
  DAUGMAN_ANGLES.push(t, Math.PI - t);
}
const DAUGMAN_COS = DAUGMAN_ANGLES.map(Math.cos), DAUGMAN_SIN = DAUGMAN_ANGLES.map(Math.sin);

function daugmanCheck(img, c, r) {
  const { data, width, height } = img;
  const gray = new Float32Array(width * height);
  for (let i = 0, j = 0; i < gray.length; i++, j += 4) gray[i] = 0.299 * data[j] + 0.587 * data[j + 1] + 0.114 * data[j + 2];
  // Clip specular highlights (catchlights) so they don't masquerade as a dark-to-bright edge.
  const cap = percentile(gray, 0.9);
  for (let i = 0; i < gray.length; i++) if (gray[i] > cap) gray[i] = cap;
  const sample = (x, y) => {
    x = clamp(x, 0, width - 1.001); y = clamp(y, 0, height - 1.001);
    const x0 = x | 0, y0 = y | 0, fx = x - x0, fy = y - y0, i = y0 * width + x0;
    return (gray[i] * (1 - fx) + gray[i + 1] * fx) * (1 - fy) + (gray[i + width] * (1 - fx) + gray[i + width + 1] * fx) * fy;
  };

  const radii = [];
  for (let k = 0.65; k <= 1.4501; k += 0.05) radii.push(k * r);
  const n = DAUGMAN_ANGLES.length, span = 0.8 * r, step = r / 6;
  const I = new Float32Array(radii.length);
  let best = null;
  for (let dy = -span; dy <= span + 1e-6; dy += step) {
    for (let dx = -span; dx <= span + 1e-6; dx += step) {
      const cx = c.x + dx, cy = c.y + dy;
      for (let ri = 0; ri < radii.length; ri++) {
        let sum = 0;
        for (let a = 0; a < n; a++) sum += sample(cx + radii[ri] * DAUGMAN_COS[a], cy + radii[ri] * DAUGMAN_SIN[a]);
        I[ri] = sum / n;
      }
      // Smoothed derivative: the iris is darker than the sclera, so the limbus is a positive step outward.
      // Candidates span 0.75r–1.35r: the limbus, not the pupil boundary inside it.
      for (let ri = 2; ri < radii.length - 2; ri++) {
        const d = (I[ri + 1] + I[ri + 2] - I[ri - 1] - I[ri - 2]) / 2;
        if (!best || d > best.d) best = { x: cx, y: cy, r: radii[ri], d };
      }
    }
  }
  if (!best || best.d < 3) return { none: true };
  return { ...best, err: Math.hypot(best.x - c.x, best.y - c.y) / r, ratio: best.r / r };
}

function radialContrast(img, c, r, poly) {
  const { data, width, height } = img;
  let inSum = 0, inN = 0, ringSum = 0, ringN = 0;
  const r1 = r * 0.7, r2 = r * 1.25, r3 = r * 1.9;
  const x0 = Math.max(0, Math.floor(c.x - r3)), x1 = Math.min(width - 1, Math.ceil(c.x + r3));
  const y0 = Math.max(0, Math.floor(c.y - r3)), y1 = Math.min(height - 1, Math.ceil(c.y + r3));
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const d = Math.hypot(x - c.x, y - c.y);
      if (d > r3 || (d > r1 && d < r2)) continue;
      if (!pointInPolygon(x, y, poly)) continue;
      const i = (y * width + x) * 4;
      const lum = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
      if (d <= r1) { inSum += lum; inN++; } else { ringSum += lum; ringN++; }
    }
  }
  if (inN < 10 || ringN < 15) return null;
  const inner = inSum / inN, ring = ringSum / ringN;
  return (ring - inner) / Math.max(ring, 1);
}

function houghCheck(img, c, r) {
  const cv = state.cv;
  if (!cv) return null;
  // Search a window of ±2.2r around the proposal, upsampled so the iris radius is ~24 px.
  const half = r * 2.2;
  const x0 = Math.max(0, Math.floor(c.x - half)), y0 = Math.max(0, Math.floor(c.y - half));
  const x1 = Math.min(img.width, Math.ceil(c.x + half)), y1 = Math.min(img.height, Math.ceil(c.y + half));
  if (x1 - x0 < 8 || y1 - y0 < 8) return { none: true };
  const k = 24 / r;
  let src, win, gray, circles;
  try {
    src = cv.matFromImageData(img);
    win = src.roi(new cv.Rect(x0, y0, x1 - x0, y1 - y0));
    gray = new cv.Mat();
    circles = new cv.Mat();
    cv.cvtColor(win, gray, cv.COLOR_RGBA2GRAY);
    cv.resize(gray, gray, new cv.Size(Math.round((x1 - x0) * k), Math.round((y1 - y0) * k)), 0, 0, cv.INTER_CUBIC);
    cv.equalizeHist(gray, gray);
    cv.medianBlur(gray, gray, 5);
    cv.HoughCircles(gray, circles, cv.HOUGH_GRADIENT, 1, 10, 60, 13, 16, 32);
    // Candidates are ordered by accumulator votes; only the strongest few are considered.
    const d = circles.data32F;
    let best = null;
    for (let i = 0; i < Math.min(circles.cols, 5); i++) {
      const cand = { x: d[i * 3] / k + x0, y: d[i * 3 + 1] / k + y0, r: d[i * 3 + 2] / k };
      const err = Math.hypot(cand.x - c.x, cand.y - c.y) / r, ratio = cand.r / r;
      const cost = err + Math.abs(1 - ratio);
      if (!best || cost < best.cost) best = { ...cand, err, ratio, cost };
    }
    return best ?? { none: true };
  } catch (err) {
    console.warn("Hough check failed", err);
    return null;
  } finally {
    src?.delete(); win?.delete(); gray?.delete(); circles?.delete();
  }
}

function deriveFaceMetrics(eyes, frameW, live) {
  const { right: R, left: L } = eyes;
  const earAvg = (R.ear + L.ear) / 2;

  if (live) {
    if (!session.eyeClosed && earAvg < EAR_CLOSE) session.eyeClosed = true;
    else if (session.eyeClosed && earAvg > EAR_OPEN) { session.eyeClosed = false; session.blinks++; }
  }

  let distanceCm = null;
  if (R.verdict !== "occl" && L.verdict !== "occl") {
    const diameterPx = R.fit.r + L.fit.r;
    const focalPx = (frameW / 2) / Math.tan((ASSUMED_HFOV_DEG / 2) * Math.PI / 180);
    distanceCm = (focalPx * IRIS_DIAMETER_MM / diameterPx) / 10;
  }

  const raw = { x: (R.gx + L.gx) / 2, y: (R.gy + L.gy) / 2 };
  const g = live && session.gaze ? { x: ema(session.gaze.x, raw.x, 0.35), y: ema(session.gaze.y, raw.y, 0.35) } : raw;
  session.gaze = g;
  const gaze = { x: g.x - session.gazeOffset.x, y: g.y - session.gazeOffset.y };

  for (const e of [R, L]) {
    if (e.verdict === "occl") continue;
    session.judged++;
    if (e.verdict === "pass") session.pass++;
  }

  const hist = session.hist;
  push(hist.earR, R.ear); push(hist.earL, L.ear);
  push(hist.scoreR, R.score); push(hist.scoreL, L.score);
  if (earAvg > EAR_OPEN) push(session.gazeTrail, { ...gaze }, 300);

  return { earAvg, distanceCm, gaze };
}

/* ───────────────────────── drawing ───────────────────────── */

function drawOverlay(pts, eyes, w) {
  const lw = Math.max(1.5, w / 640);
  const T = state.toggles;

  if (T.mesh) {
    ctx.fillStyle = "rgba(139, 123, 255, 0.55)";
    const s = Math.max(1.2, w / 900);
    for (const p of pts) ctx.fillRect(p.x - s / 2, p.y - s / 2, s, s);
  }

  for (const side of ["right", "left"]) {
    const e = eyes[side], def = EYES[side];
    if (T.contours) {
      ctx.strokeStyle = "rgba(230, 237, 247, 0.75)";
      ctx.lineWidth = lw;
      polyline(ctx, def.contour.map((i) => pts[i]), true);
    }
    if (T.validators) {
      for (const [m, color, dash] of [[e.daugman, COLORS.violet, [lw, lw * 2]], [e.hough, COLORS.pink, [lw * 3, lw * 3]]]) {
        if (!m || m.none) continue;
        ctx.setLineDash(dash);
        ctx.strokeStyle = color;
        ctx.lineWidth = lw;
        circle(ctx, m.x, m.y, m.r);
      }
      ctx.setLineDash([]);
    }
    if (T.iris) {
      const col = verdictColor(e.verdict);
      ctx.strokeStyle = col;
      ctx.lineWidth = lw * 1.4;
      ctx.shadowColor = col; ctx.shadowBlur = lw * 6;
      circle(ctx, e.fit.x, e.fit.y, e.fit.r);
      ctx.shadowBlur = 0;
      crosshair(ctx, e.fit.x, e.fit.y, e.fit.r * 0.35, lw, col);
      ctx.fillStyle = "#fff";
      for (const i of def.iris) dot(ctx, pts[i].x, pts[i].y, lw * 1.3);
    }
    if (T.gaze && e.verdict !== "occl") {
      // Gaze is expressed in subject coordinates; image x runs opposite to the subject's right.
      const k = e.eyeW * 1.6;
      const g = state.lastAnalysis?.gaze ?? { x: e.gx, y: e.gy };
      arrow(ctx, e.fit.x, e.fit.y, e.fit.x - g.x * k, e.fit.y + g.y * k, lw * 1.3, COLORS.amber);
    }
  }
}

function drawCropOverlay(side, e, pts) {
  const c = cropCtx[side], cw = crops[side].width, ch = crops[side].height;
  const sx = cw / e.roi.w, sy = ch / e.roi.h;
  const map = (p) => ({ x: (p.x - e.roi.x) * sx, y: (p.y - e.roi.y) * sy });
  const f = map(e.fit);

  c.lineWidth = 1.5;
  c.strokeStyle = "rgba(230,237,247,.55)";
  polyline(c, EYES[side].contour.map((i) => map(pts[i])), true);

  for (const [m, color, dash] of [[e.daugman, COLORS.violet, [2, 4]], [e.hough, COLORS.pink, [5, 4]]]) {
    if (!m || m.none) continue;
    const mp = map(m);
    c.setLineDash(dash);
    c.strokeStyle = color;
    c.lineWidth = 2;
    circle(c, mp.x, mp.y, m.r * sx);
  }
  c.setLineDash([]);
  const col = verdictColor(e.verdict);
  c.strokeStyle = col;
  c.lineWidth = 2.5;
  c.shadowColor = col; c.shadowBlur = 10;
  circle(c, f.x, f.y, e.fit.r * sx);
  c.shadowBlur = 0;
  crosshair(c, f.x, f.y, e.fit.r * sx * 0.4, 1.5, col);
}

function drawEmptyCrop(side) {
  const c = cropCtx[side], { width, height } = crops[side];
  c.fillStyle = "#04060a";
  c.fillRect(0, 0, width, height);
  c.fillStyle = COLORS.muted;
  c.font = "14px 'JetBrains Mono', monospace";
  c.textAlign = "center";
  c.save();
  if (state.source === "camera") { c.translate(width, 0); c.scale(-1, 1); }
  c.fillText("no face detected", width / 2, height / 2 + 5);
  c.restore();
}

function drawRoi(target, src, roi, W, H) {
  // Clip the source rectangle to the frame and map the visible part to the matching destination rect.
  const sx0 = Math.max(0, roi.x), sy0 = Math.max(0, roi.y);
  const sx1 = Math.min(src.width, roi.x + roi.w), sy1 = Math.min(src.height, roi.y + roi.h);
  target.fillStyle = "#04060a";
  target.fillRect(0, 0, W, H);
  if (sx1 <= sx0 || sy1 <= sy0) return;
  const kx = W / roi.w, ky = H / roi.h;
  target.drawImage(src, sx0, sy0, sx1 - sx0, sy1 - sy0,
    (sx0 - roi.x) * kx, (sy0 - roi.y) * ky, (sx1 - sx0) * kx, (sy1 - sy0) * ky);
}

function drawChart(el, seriesA, seriesB, { min, max, threshold, area = false }) {
  const c = el.getContext("2d"), W = el.width, H = el.height, pad = 6;
  c.clearRect(0, 0, W, H);
  c.strokeStyle = COLORS.line; c.lineWidth = 1;
  for (let i = 1; i < 4; i++) { const y = (H / 4) * i; c.beginPath(); c.moveTo(0, y); c.lineTo(W, y); c.stroke(); }
  const yOf = (v) => H - pad - ((clamp((v - min) / (max - min))) * (H - pad * 2));
  if (threshold != null) {
    c.setLineDash([6, 6]); c.strokeStyle = COLORS.amber;
    c.beginPath(); c.moveTo(0, yOf(threshold)); c.lineTo(W, yOf(threshold)); c.stroke(); c.setLineDash([]);
  }
  const plot = (data, color) => {
    if (data.length < 2) return;
    const step = W / (HISTORY - 1), off = W - (data.length - 1) * step;
    c.beginPath();
    data.forEach((v, i) => (i ? c.lineTo(off + i * step, yOf(v)) : c.moveTo(off, yOf(v))));
    c.strokeStyle = color; c.lineWidth = 2; c.stroke();
    if (area) {
      c.lineTo(W, H); c.lineTo(off, H); c.closePath();
      const g = c.createLinearGradient(0, 0, 0, H);
      g.addColorStop(0, color + "33"); g.addColorStop(1, color + "00");
      c.fillStyle = g; c.fill();
    }
  };
  plot(seriesB, COLORS.violet);
  plot(seriesA, COLORS.cyan);
}

function drawGazeMap() {
  const el = $("gazeMap"), c = el.getContext("2d"), W = el.width, R = W / 2 - 16, C = W / 2;
  c.clearRect(0, 0, W, W);
  c.strokeStyle = COLORS.line; c.lineWidth = 1;
  for (const k of [0.33, 0.66, 1]) circle(c, C, C, R * k);
  c.beginPath(); c.moveTo(C - R, C); c.lineTo(C + R, C); c.moveTo(C, C - R); c.lineTo(C, C + R); c.stroke();
  c.fillStyle = COLORS.muted; c.font = "11px 'JetBrains Mono', monospace"; c.textAlign = "center";
  c.fillText("UP", C, 12); c.fillText("DOWN", C, W - 3);
  c.textAlign = "left"; c.fillText("L", 2, C - 4); c.textAlign = "right"; c.fillText("R", W - 2, C - 4);

  const range = 0.45;
  const toXY = (g) => ({ x: C + clamp(g.x / range, -1, 1) * R, y: C + clamp(g.y / (range * 1.4), -1, 1) * R });
  const trail = session.gazeTrail;
  trail.forEach((g, i) => {
    const p = toXY(g);
    c.fillStyle = `rgba(63, 224, 208, ${0.05 + 0.35 * (i / trail.length)})`;
    dot(c, p.x, p.y, 3);
  });
  const cur = state.lastAnalysis?.gaze;
  if (cur && state.lastAnalysis.earAvg > EAR_OCCLUDED) {
    const p = toXY(cur);
    c.shadowColor = COLORS.amber; c.shadowBlur = 16; c.fillStyle = COLORS.amber;
    dot(c, p.x, p.y, 7); c.shadowBlur = 0;
  }
}

/* ───────────────────────── ui ───────────────────────── */

function updateUi(a, live, noFace) {
  const now = performance.now();
  if (live && now - state.lastUiAt < 80) return;
  state.lastUiAt = now;

  setChip("chipFace", noFace ? "" : "ok", noFace ? "No face" : "Face locked");
  $("chipFps").textContent = live ? `${state.fps.toFixed(0)} fps` : "still image";
  $("mFrames").textContent = session.frames.toLocaleString();
  $("mLatency").textContent = `${state.latency.toFixed(1)} ms / frame`;

  if (noFace || !a) {
    for (const side of ["right", "left"]) {
      setVerdict(side, null);
      for (const k of ["c", "r", "rms", "d", "h", "k"]) $(`${k}-${side}`).textContent = "—";
      $(`m-${side}`).style.width = "0";
    }
    if (!live) {
      for (const id of ["mDistance", "mGaze", "mOpen", "mPass"]) $(id).textContent = "—";
    }
    drawGazeMap();
    return;
  }

  for (const side of ["right", "left"]) {
    const e = a.eyes[side];
    setVerdict(side, e);
    $(`c-${side}`).textContent = `${e.fit.x.toFixed(0)}, ${e.fit.y.toFixed(0)}`;
    $(`r-${side}`).textContent = `${e.fit.r.toFixed(2)} px`;
    $(`rms-${side}`).textContent = Number.isFinite(e.fit.rms) ? `${e.fit.rms.toFixed(2)} px` : "—";
    $(`d-${side}`).textContent = fmtDelta(e.daugman);
    $(`h-${side}`).textContent = fmtDelta(e.hough);
    $(`k-${side}`).textContent = e.contrast == null ? "—" : `${(e.contrast * 100).toFixed(0)}%`;
    const m = $(`m-${side}`);
    m.style.width = `${(e.score * 100).toFixed(0)}%`;
    m.style.background = verdictColor(e.verdict);
  }

  $("mDistance").textContent = a.distanceCm ? `${a.distanceCm.toFixed(1)} cm` : "—";
  $("mGaze").textContent = gazeLabel(a.gaze, a.earAvg);
  $("mGazeSub").textContent = `x ${fmtSigned(a.gaze.x)} · y ${fmtSigned(a.gaze.y)}`;
  $("mOpen").textContent = a.earAvg.toFixed(3);
  $("mBlinks").textContent = session.blinks;
  const minutes = (now - session.startedAt) / 60000;
  $("mBlinkRate").textContent = live && minutes > 0.1 ? `${(session.blinks / minutes).toFixed(1)} / min` : "— / min";
  $("mPass").textContent = session.judged ? `${((session.pass / session.judged) * 100).toFixed(0)}%` : "—";

  drawChart($("chartEar"), session.hist.earR, session.hist.earL, { min: 0.05, max: 0.45, threshold: EAR_CLOSE });
  drawChart($("chartScore"), session.hist.scoreR, session.hist.scoreL, { min: 0, max: 1, threshold: PASS_THRESHOLD, area: true });
  drawGazeMap();
}

function setVerdict(side, e) {
  const el = $(`verdict-${side}`);
  el.className = "verdict" + (e ? ` ${e.verdict}` : "");
  el.textContent = e ? `${{ pass: "PASS", fail: "FAIL", occl: "OCCLUDED" }[e.verdict]} · ${e.score.toFixed(2)}` : "—";
  el.title = e ? e.reason : "";
}

function gazeLabel(g, ear) {
  if (ear < EAR_OCCLUDED) return "eyes closed";
  const h = g.x > 0.1 ? "Right" : g.x < -0.1 ? "Left" : "";
  const v = g.y < -0.18 ? "Up" : g.y > 0.22 ? "Down" : "";
  return [v, h].filter(Boolean).join("-") || "Center";
}

function setChip(id, cls, text) {
  const el = $(id);
  el.className = `chip ${cls}`;
  el.lastChild.textContent = text;
}

function setMirrored(on) {
  stage.classList.toggle("mirrored", on);
  eyeRow.classList.toggle("mirrored", on);
}

function setActionsEnabled(on) {
  $("btnCapture").disabled = !on;
  $("btnReport").disabled = !on;
}

let toastTimer;
function toast(msg, isErr = false) {
  const el = $("toast");
  el.textContent = msg;
  el.className = `toast show${isErr ? " err" : ""}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (el.className = "toast"), 3600);
}

function download(blob, name) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

function reportDetector(m) {
  if (m == null) return "unavailable";
  if (m.none) return "no circle";
  return { center_error_r: round(m.err, 3), radius_ratio: round(m.ratio, 3) };
}

function buildReport() {
  const a = state.lastAnalysis;
  const eye = (e) => e && {
    verdict: e.verdict, reason: e.reason, score: round(e.score, 3),
    center_px: { x: round(e.fit.x, 2), y: round(e.fit.y, 2) }, radius_px: round(e.fit.r, 3),
    fit_rms_px: Number.isFinite(e.fit.rms) ? round(e.fit.rms, 3) : null,
    ear: round(e.ear, 4), contrast: e.contrast == null ? null : round(e.contrast, 4),
    daugman: reportDetector(e.daugman), hough: reportDetector(e.hough),
  };
  return {
    generated_at: new Date().toISOString(),
    source: state.source,
    frame: a ? { width: a.w, height: a.h } : null,
    session: {
      frames: session.frames, blinks: session.blinks,
      duration_s: round((performance.now() - session.startedAt) / 1000, 1),
      pass_rate: session.judged ? round(session.pass / session.judged, 3) : null,
    },
    face_detected: !!a,
    estimated_distance_cm: a?.distanceCm ? round(a.distanceCm, 1) : null,
    gaze: a ? { x: round(a.gaze.x, 3), y: round(a.gaze.y, 3), label: gazeLabel(a.gaze, a.earAvg) } : null,
    eyes: a ? { right: eye(a.eyes.right), left: eye(a.eyes.left) } : null,
  };
}

/* ───────────────────────── events ───────────────────────── */

$("btnCamera").addEventListener("click", () => (state.source === "camera" ? stopCamera() : startCamera()));
$("btnSample").addEventListener("click", () => analyzeImageUrl("assets/sample.jpg"));
$("fileInput").addEventListener("change", (e) => { analyzeImageFile(e.target.files[0]); e.target.value = ""; });
$("btnCapture").addEventListener("click", () => canvas.toBlob((b) => b && download(b, `irisvision-${Date.now()}.png`), "image/png"));
$("btnReport").addEventListener("click", () =>
  download(new Blob([JSON.stringify(buildReport(), null, 2)], { type: "application/json" }), `irisvision-report-${Date.now()}.json`));
$("btnResetGaze").addEventListener("click", () => {
  if (session.gaze) session.gazeOffset = { ...session.gaze };
  session.gazeTrail = [];
  drawGazeMap();
  toast("Gaze recentred — look at the screen centre while resetting.");
});

document.querySelectorAll("[data-toggle]").forEach((el) => {
  el.addEventListener("change", () => { state.toggles[el.dataset.toggle] = el.checked; rerenderImage(); });
});

["dragenter", "dragover"].forEach((t) => stage.addEventListener(t, (e) => { e.preventDefault(); stage.classList.add("dragging"); }));
["dragleave", "drop"].forEach((t) => stage.addEventListener(t, (e) => { e.preventDefault(); stage.classList.remove("dragging"); }));
stage.addEventListener("drop", (e) => analyzeImageFile(e.dataTransfer.files[0]));
document.addEventListener("visibilitychange", () => { if (document.hidden && state.source === "camera") stopCamera(); });

for (const side of ["right", "left"]) drawEmptyCrop(side);
drawGazeMap();
if (new URLSearchParams(location.search).has("sample")) analyzeImageUrl("assets/sample.jpg");

/* ───────────────────────── math helpers ───────────────────────── */

// Centred algebraic (Kåsa) least-squares circle fit.
function fitCircle(points) {
  const n = points.length;
  if (n < 3) return null;
  const mx = points.reduce((s, p) => s + p.x, 0) / n, my = points.reduce((s, p) => s + p.y, 0) / n;
  let Suu = 0, Svv = 0, Suv = 0, Suuu = 0, Svvv = 0, Suvv = 0, Svuu = 0;
  for (const p of points) {
    const u = p.x - mx, v = p.y - my;
    Suu += u * u; Svv += v * v; Suv += u * v;
    Suuu += u * u * u; Svvv += v * v * v; Suvv += u * v * v; Svuu += v * u * u;
  }
  const det = Suu * Svv - Suv * Suv;
  if (Math.abs(det) < 1e-9) return null;
  const b1 = 0.5 * (Suuu + Suvv), b2 = 0.5 * (Svvv + Svuu);
  const uc = (b1 * Svv - b2 * Suv) / det, vc = (Suu * b2 - Suv * b1) / det;
  const r = Math.sqrt(uc * uc + vc * vc + (Suu + Svv) / n);
  const x = uc + mx, y = vc + my;
  const rms = Math.sqrt(points.reduce((s, p) => s + (Math.hypot(p.x - x, p.y - y) - r) ** 2, 0) / n);
  return { x, y, r, rms };
}

function project(p, a, b) {
  const dx = b.x - a.x, dy = b.y - a.y;
  return ((p.x - a.x) * dx + (p.y - a.y) * dy) / (dx * dx + dy * dy || 1);
}

function pointInPolygon(x, y, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const a = poly[i], b = poly[j];
    if ((a.y > y) !== (b.y > y) && x < ((b.x - a.x) * (y - a.y)) / (b.y - a.y) + a.x) inside = !inside;
  }
  return inside;
}

function percentile(arr, q) {
  const sorted = Float32Array.from(arr).sort();
  return sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
}
function dist(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }
function clamp(v, lo = 0, hi = 1) { return Math.min(hi, Math.max(lo, v)); }
function ema(prev, next, k) { return prev ? prev + (next - prev) * k : next; }
function round(v, d) { const k = 10 ** d; return Math.round(v * k) / k; }
function push(arr, v, max = HISTORY) { arr.push(v); if (arr.length > max) arr.shift(); }
function fmtDelta(m) { return m == null ? "n/a" : m.none ? "none" : `${m.err.toFixed(2)}r · ${m.ratio.toFixed(2)}×`; }
function fmtSigned(v) { return (v >= 0 ? "+" : "") + v.toFixed(2); }
function verdictColor(v) { return v === "pass" ? COLORS.green : v === "occl" ? COLORS.amber : v === "fail" ? COLORS.red : COLORS.cyan; }

function circle(c, x, y, r) { c.beginPath(); c.arc(x, y, Math.max(0.5, r), 0, Math.PI * 2); c.stroke(); }
function dot(c, x, y, r) { c.beginPath(); c.arc(x, y, r, 0, Math.PI * 2); c.fill(); }
function polyline(c, pts, close) {
  c.beginPath();
  pts.forEach((p, i) => (i ? c.lineTo(p.x, p.y) : c.moveTo(p.x, p.y)));
  if (close) c.closePath();
  c.stroke();
}
function crosshair(c, x, y, s, lw, color) {
  c.strokeStyle = color; c.lineWidth = lw;
  c.beginPath(); c.moveTo(x - s, y); c.lineTo(x + s, y); c.moveTo(x, y - s); c.lineTo(x, y + s); c.stroke();
}
function arrow(c, x0, y0, x1, y1, lw, color) {
  const ang = Math.atan2(y1 - y0, x1 - x0), head = lw * 4;
  c.strokeStyle = color; c.fillStyle = color; c.lineWidth = lw;
  c.beginPath(); c.moveTo(x0, y0); c.lineTo(x1, y1); c.stroke();
  c.beginPath(); c.moveTo(x1, y1);
  c.lineTo(x1 - head * Math.cos(ang - 0.45), y1 - head * Math.sin(ang - 0.45));
  c.lineTo(x1 - head * Math.cos(ang + 0.45), y1 - head * Math.sin(ang + 0.45));
  c.closePath(); c.fill();
}
