const CANVAS_SIZE = 280;
const STROKE_COLOR = '#ffffff'; // white stroke on black background like MNIST
const STROKE_WIDTH = 22;
const FLIP_INPUT_HORIZONTAL = true; // Flip 28x28 input to match EMNIST orientation
const LABELS = [
  'a','b','c','d','e','f','g','h','i','j',
  'k','l','m','n','o','p','q','r','s','t',
  'u','v','w','x','y','z'
];
let submitDelayMs = 300; // adjustable via slider

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
ctx.fillStyle = '#000000';
ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
ctx.lineCap = 'round';
ctx.lineJoin = 'round';
ctx.strokeStyle = STROKE_COLOR;
ctx.lineWidth = STROKE_WIDTH;

let drawing = false;
let model = null;
let outputStr = '';
let submitTimer = null;
let strokeDrawn = false; // true only if the last stroke had movement

function getPos(e) {
  const rect = canvas.getBoundingClientRect();
  if (e.touches && e.touches[0]) {
    return { x: e.touches[0].clientX - rect.left, y: e.touches[0].clientY - rect.top };
  }
  return { x: e.clientX - rect.left, y: e.clientY - rect.top };
}

function startDraw(e) {
  drawing = true;
  if (submitTimer) {
    clearTimeout(submitTimer);
    submitTimer = null;
  }
  strokeDrawn = false;
  const { x, y } = getPos(e);
  ctx.beginPath();
  ctx.moveTo(x, y);
}

function draw(e) {
  if (!drawing) return;
  const { x, y } = getPos(e);
  ctx.lineTo(x, y);
  ctx.stroke();
  strokeDrawn = true;
}

function endDraw() {
  // Only act if a stroke was in progress
  if (!drawing) return;
  drawing = false;
  ctx.closePath();
  // start inactivity timer for auto-submit
  if (!strokeDrawn) return; // ignore simple clicks without drawing
  if (!hasInk(canvas)) return; // ensure there is visible ink on the canvas
  if (submitTimer) clearTimeout(submitTimer);
  submitTimer = setTimeout(() => {
    // Double-check ink before submitting
    if (hasInk(canvas)) {
      predictAndHandle({ append: true, clearAfter: true });
    }
    submitTimer = null;
  }, submitDelayMs);
}

canvas.addEventListener('mousedown', startDraw);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', endDraw);
canvas.addEventListener('mouseleave', endDraw);
canvas.addEventListener('touchstart', (e) => { e.preventDefault(); startDraw(e); });
canvas.addEventListener('touchmove', (e) => { e.preventDefault(); draw(e); });
canvas.addEventListener('touchend', (e) => { e.preventDefault(); endDraw(e); });

// Right-click to predict, append digit, and clear canvas for next digit (manual override)
canvas.addEventListener('contextmenu', async (e) => {
  e.preventDefault();
  if (submitTimer) { clearTimeout(submitTimer); submitTimer = null; }
  if (hasInk(canvas)) {
    await predictAndHandle({ append: true, clearAfter: true });
  }
});

async function ensureModel() {
  if (!model) {
    // Serve from repo root so this absolute path works
    model = await tf.loadLayersModel('/model/model.json');
  }
}

async function predictAndHandle({ append, clearAfter }) {
  try {
    await ensureModel();
    const input = tf.tidy(() => {
      const img = preprocessCanvasTo28x28(canvas);
      return img.expandDims(0); // [1, 28, 28, 1]
    });
    const logits = model.predict(input);
    const probs = logits.dataSync();
    input.dispose();
    tf.dispose(logits);
    let idx = argmaxIndex(probs);
    // Ambiguity helpers for b vs d and p vs q
    idx = adjustAmbiguitiesLetters(idx, probs, canvas);
    showResult(probs, idx);
    if (append) appendDigit(LABELS[idx]);
    if (clearAfter) { clearCanvas(); strokeDrawn = false; }
  } catch (err) {
    console.error(err);
    alert('Prediction failed. Make sure /model/model.json exists and you are serving from repo root.');
  }
}

function argmaxIndex(arr) {
  let idx = 0, val = arr[0];
  for (let i = 1; i < arr.length; i++) {
    if (arr[i] > val) { val = arr[i]; idx = i; }
  }
  return idx;
}

function appendDigit(ch) {
  outputStr += String(ch);
  const outEl = document.getElementById('outputValue');
  if (outEl) outEl.textContent = outputStr.length ? outputStr : '(empty)';
}

function clearCanvas() {
  ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  ctx.fillStyle = '#000000';
  ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  ctx.fillStyle = '#000000';
}

function hasInk(cnv) {
  const w = cnv.width, h = cnv.height;
  const data = cnv.getContext('2d').getImageData(0, 0, w, h).data;
  const threshold = 10; // brightness > threshold counts as ink
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i], g = data[i + 1], b = data[i + 2];
    const val = (r + g + b) / 3;
    if (val > threshold) return true;
  }
  return false;
}

// Output editing controls
function updateOutputDisplay() {
  const outEl = document.getElementById('outputValue');
  if (outEl) outEl.textContent = outputStr.length ? outputStr : '(empty)';
}

const backspaceBtn = document.getElementById('backspaceBtn');
if (backspaceBtn) {
  backspaceBtn.addEventListener('click', () => {
    if (outputStr.length > 0) {
      outputStr = outputStr.slice(0, -1);
      updateOutputDisplay();
    }
  });
}

const clearOutputBtn = document.getElementById('clearOutputBtn');
if (clearOutputBtn) {
  clearOutputBtn.addEventListener('click', () => {
    outputStr = '';
    updateOutputDisplay();
  });
}

// Delay slider wiring
const delaySlider = document.getElementById('delaySlider');
const delayValue = document.getElementById('delayValue');
if (delaySlider && delayValue) {
  const setDelay = (v) => {
    const n = Math.max(0, Math.min(1000, parseInt(v, 10) || 0));
    submitDelayMs = n;
    delayValue.textContent = String(n);
  };
  setDelay(delaySlider.value || '300');
  delaySlider.addEventListener('input', (e) => {
    setDelay(e.target.value);
    // If a timer is running, reschedule with new delay from now
    if (submitTimer) {
      clearTimeout(submitTimer);
      submitTimer = setTimeout(() => {
        if (hasInk(canvas)) {
          predictAndHandle({ append: true, clearAfter: true });
        }
        submitTimer = null;
      }, submitDelayMs);
    }
  });
}

function preprocessCanvasTo28x28(srcCanvas) {
  // Get bounding box of non-black pixels
  const srcCtx = srcCanvas.getContext('2d');
  const img = srcCtx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  let minX = CANVAS_SIZE, minY = CANVAS_SIZE, maxX = 0, maxY = 0;
  const threshold = 10; // pixel brightness threshold
  for (let y = 0; y < CANVAS_SIZE; y++) {
    for (let x = 0; x < CANVAS_SIZE; x++) {
      const i = (y * CANVAS_SIZE + x) * 4;
      const r = img.data[i], g = img.data[i + 1], b = img.data[i + 2];
      const val = (r + g + b) / 3;
      if (val > threshold) {
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (minX > maxX || minY > maxY) {
    // Nothing drawn; just scale whole canvas
    minX = 0; minY = 0; maxX = CANVAS_SIZE - 1; maxY = CANVAS_SIZE - 1;
  }

  const bw = maxX - minX + 1;
  const bh = maxY - minY + 1;
  const boxSize = Math.max(bw, bh);

  // Create a square crop around the bounding box
  const tmpCrop = document.createElement('canvas');
  tmpCrop.width = boxSize;
  tmpCrop.height = boxSize;
  const cropCtx = tmpCrop.getContext('2d');
  cropCtx.fillStyle = '#000';
  cropCtx.fillRect(0, 0, boxSize, boxSize);

  const dx = (boxSize - bw) / 2;
  const dy = (boxSize - bh) / 2;
  cropCtx.drawImage(srcCanvas, minX, minY, bw, bh, dx, dy, bw, bh);

  // Scale to 20x20 and place centered into 28x28 per common MNIST preprocessing
  const target = document.createElement('canvas');
  target.width = 28;
  target.height = 28;
  const tctx = target.getContext('2d');
  tctx.fillStyle = '#000';
  tctx.fillRect(0, 0, 28, 28);

  const scale = 20 / boxSize;
  const w = Math.max(1, Math.round(boxSize * scale));
  const h = Math.max(1, Math.round(boxSize * scale));
  const sx = Math.floor((28 - w) / 2);
  const sy = Math.floor((28 - h) / 2);
  tctx.imageSmoothingEnabled = true;
  tctx.imageSmoothingQuality = 'high';
  tctx.drawImage(tmpCrop, 0, 0, boxSize, boxSize, sx, sy, w, h);

  // Convert to grayscale float32 tensor scaled [0,1], shape [28,28,1]
  const imgData = tctx.getImageData(0, 0, 28, 28);
  const buf = new Float32Array(28 * 28);
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      const srcX = FLIP_INPUT_HORIZONTAL ? (27 - x) : x; // mirror horizontally if enabled
      const idx = (y * 28 + srcX) * 4;
      const r = imgData.data[idx];
      const g = imgData.data[idx + 1];
      const b = imgData.data[idx + 2];
      buf[y * 28 + x] = (r + g + b) / (3 * 255);
    }
  }
  return tf.tensor(buf, [28, 28, 1]);
}

function showResult(probs, topIdxOverride = null) {
  // Prepare sorted probabilities descending by confidence
  const rows = [];
  for (let d = 0; d < probs.length; d++) rows.push({ d, p: probs[d] });
  rows.sort((a, b) => b.p - a.p);

  // Top-1 digit
  const top = rows[0];
  const topIdx = topIdxOverride != null ? topIdxOverride : top.d;
  document.getElementById('predDigit').textContent = `${LABELS[topIdx]}`;

  // Render bars in sorted order (highest at top)
  const probsEl = document.getElementById('probs');
  probsEl.innerHTML = '';
  for (const { d, p } of rows) {
    const pct = Math.round(p * 1000) / 10; // 1 decimal
    if (pct <= 0) continue; // only show predictions over 0.0%
    const row = document.createElement('div');
    row.className = 'bar';
    row.innerHTML = `
      <div>${LABELS[d]}</div>
      <div class="track"><div class="fill" style="width:${Math.min(100, pct)}%"></div></div>
      <div class="pct">${pct.toFixed(1)}%</div>
    `;
    probsEl.appendChild(row);
  }
}

function getInkBoundingBox(srcCanvas) {
  const ctx2 = srcCanvas.getContext('2d');
  const img = ctx2.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  let minX = CANVAS_SIZE, minY = CANVAS_SIZE, maxX = -1, maxY = -1;
  const threshold = 10;
  for (let y = 0; y < CANVAS_SIZE; y++) {
    for (let x = 0; x < CANVAS_SIZE; x++) {
      const i = (y * CANVAS_SIZE + x) * 4;
      const r = img.data[i], g = img.data[i + 1], b = img.data[i + 2];
      const val = (r + g + b) / 3;
      if (val > threshold) {
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < 0 || maxY < 0) return null;
  return { x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1 };
}

function leftColumnCoverage(srcCanvas, frac = 0.15) {
  const ctx2 = srcCanvas.getContext('2d');
  const w = CANVAS_SIZE, h = CANVAS_SIZE;
  const img = ctx2.getImageData(0, 0, w, h);
  const colLimit = Math.floor(w * frac);
  const threshold = 10;
  let rowsWithInkLeft = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < colLimit; x++) {
      const i = (y * w + x) * 4;
      const r = img.data[i], g = img.data[i + 1], b = img.data[i + 2];
      const val = (r + g + b) / 3;
      if (val > threshold) { rowsWithInkLeft++; break; }
    }
  }
  return rowsWithInkLeft / h; // 0..1
}

function getTopKIndices(probs, k) {
  const idxs = probs.map((p, i) => i).sort((a, b) => probs[b] - probs[a]);
  return idxs.slice(0, k);
}

function rightColumnCoverage(srcCanvas, frac = 0.15) {
  const ctx2 = srcCanvas.getContext('2d');
  const w = CANVAS_SIZE, h = CANVAS_SIZE;
  const img = ctx2.getImageData(0, 0, w, h);
  const start = Math.floor(w * (1 - frac));
  const threshold = 10;
  let rowsWithInkRight = 0;
  for (let y = 0; y < h; y++) {
    for (let x = start; x < w; x++) {
      const i = (y * w + x) * 4;
      const r = img.data[i], g = img.data[i + 1], b = img.data[i + 2];
      const val = (r + g + b) / 3;
      if (val > threshold) { rowsWithInkRight++; break; }
    }
  }
  return rowsWithInkRight / h; // 0..1
}

function adjustAmbiguitiesLetters(idx, probs, srcCanvas) {
  const top2 = getTopKIndices(probs, 2);
  const labelAt = (i) => LABELS[i];
  const setHas = (arr, ch) => arr.some((i) => labelAt(i) === ch);

  // b vs d: left stem vs right stem
  if (setHas(top2, 'b') && setHas(top2, 'd')) {
    const left = leftColumnCoverage(srcCanvas, 0.12);
    const right = rightColumnCoverage(srcCanvas, 0.12);
    if (left > right + 0.15) return LABELS.indexOf('b');
    if (right > left + 0.15) return LABELS.indexOf('d');
  }
  // p vs q: left stem vs right stem
  if (setHas(top2, 'p') && setHas(top2, 'q')) {
    const left = leftColumnCoverage(srcCanvas, 0.12);
    const right = rightColumnCoverage(srcCanvas, 0.12);
    if (left > right + 0.15) return LABELS.indexOf('p');
    if (right > left + 0.15) return LABELS.indexOf('q');
  }
  return idx;
}
