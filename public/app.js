const CANVAS_SIZE = 280;
const STROKE_COLOR = '#ffffff'; // white stroke on black background like MNIST
const STROKE_WIDTH = 22;

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

function getPos(e) {
  const rect = canvas.getBoundingClientRect();
  if (e.touches && e.touches[0]) {
    return { x: e.touches[0].clientX - rect.left, y: e.touches[0].clientY - rect.top };
  }
  return { x: e.clientX - rect.left, y: e.clientY - rect.top };
}

function startDraw(e) {
  drawing = true;
  const { x, y } = getPos(e);
  ctx.beginPath();
  ctx.moveTo(x, y);
}

function draw(e) {
  if (!drawing) return;
  const { x, y } = getPos(e);
  ctx.lineTo(x, y);
  ctx.stroke();
}

function endDraw() {
  drawing = false;
  ctx.closePath();
}

canvas.addEventListener('mousedown', startDraw);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', endDraw);
canvas.addEventListener('mouseleave', endDraw);
canvas.addEventListener('touchstart', (e) => { e.preventDefault(); startDraw(e); });
canvas.addEventListener('touchmove', (e) => { e.preventDefault(); draw(e); });
canvas.addEventListener('touchend', (e) => { e.preventDefault(); endDraw(e); });

// Right-click to predict, append digit, and clear canvas for next digit
canvas.addEventListener('contextmenu', async (e) => {
  e.preventDefault();
  await predictAndHandle({ append: true, clearAfter: true });
});

document.getElementById('clearBtn').addEventListener('click', () => {
  clearCanvas();
  document.getElementById('predDigit').textContent = '–';
  document.getElementById('probs').innerHTML = '';
});

document.getElementById('predictBtn').addEventListener('click', async () => {
  // Predict only, do not append or clear
  await predictAndHandle({ append: false, clearAfter: false });
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
    const digit = argmaxIndex(probs);
    showResult(probs);
    if (append) appendDigit(digit);
    if (clearAfter) clearCanvas();
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

function appendDigit(d) {
  outputStr += String(d);
  const outEl = document.getElementById('outputValue');
  if (outEl) outEl.textContent = outputStr.length ? outputStr : '(empty)';
}

function clearCanvas() {
  ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  ctx.fillStyle = '#000000';
  ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  ctx.fillStyle = '#000000';
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
  for (let i = 0; i < 28 * 28; i++) {
    const r = imgData.data[i * 4];
    const g = imgData.data[i * 4 + 1];
    const b = imgData.data[i * 4 + 2];
    buf[i] = (r + g + b) / (3 * 255); // already white on black, matches MNIST
  }
  return tf.tensor(buf, [28, 28, 1]);
}

function showResult(probs) {
  // Find top-1
  let bestIdx = 0, bestVal = probs[0];
  for (let i = 1; i < 10; i++) {
    if (probs[i] > bestVal) { bestVal = probs[i]; bestIdx = i; }
  }
  document.getElementById('predDigit').textContent = `${bestIdx}`;

  // Show bars for all digits
  const probsEl = document.getElementById('probs');
  probsEl.innerHTML = '';
  for (let d = 0; d < 10; d++) {
    const pct = Math.round(probs[d] * 1000) / 10; // 1 decimal
    const row = document.createElement('div');
    row.className = 'bar';
    row.innerHTML = `
      <div>${d}</div>
      <div class="track"><div class="fill" style="width:${Math.min(100, pct)}%"></div></div>
      <div class="pct">${pct.toFixed(1)}%</div>
    `;
    probsEl.appendChild(row);
  }
}
