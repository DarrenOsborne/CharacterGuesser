const CANVAS_SIZE = 280;
const STROKE_COLOR = '#ffffff';
const STROKE_WIDTH = 22;
const FLIP_INPUT_HORIZONTAL = true;
const LABELS = [
  'a','b','c','d','e','f','g','h','i','j',
  'k','l','m','n','o','p','q','r','s','t',
  'u','v','w','x','y','z'
];

let submitDelayMs = 300; // adjustable

// Canvas setup
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
ctx.fillStyle = '#000';
ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
ctx.lineCap = 'round';
ctx.lineJoin = 'round';
ctx.strokeStyle = STROKE_COLOR;
ctx.lineWidth = STROKE_WIDTH;

// State
let drawing = false;
let lastPos = null; // track last pointer position for click dots
let model = null;
let submitTimer = null;
let strokeDrawn = false;

const game = {
  target: '',
  index: 0,
  started: false,
  startMs: 0,
  endMs: 0,
  lastWrong: false,
  timerId: null,
  logs: [], // {target, predIdx, pCorrect, pOtherMax, correct}
};

// Drawing handlers
function getPos(e) {
  const rect = canvas.getBoundingClientRect();
  if (e.touches && e.touches[0]) return { x: e.touches[0].clientX - rect.left, y: e.touches[0].clientY - rect.top };
  return { x: e.clientX - rect.left, y: e.clientY - rect.top };
}
function startDraw(e) {
  drawing = true;
  if (submitTimer) { clearTimeout(submitTimer); submitTimer = null; }
  strokeDrawn = false;
  const {x,y} = getPos(e); lastPos = {x,y}; ctx.beginPath(); ctx.moveTo(x,y);
}
function draw(e) { if (!drawing) return; const {x,y}=getPos(e); lastPos = {x,y}; ctx.lineTo(x,y); ctx.stroke(); strokeDrawn=true; }
function endDraw(e) {
  if (!drawing) return;
  drawing = false; ctx.closePath();
  // If it was just a click (no move), draw a dot at the last position
  if (!strokeDrawn && lastPos){
    ctx.save();
    ctx.beginPath();
    ctx.fillStyle = STROKE_COLOR;
    ctx.arc(lastPos.x, lastPos.y, Math.max(1, STROKE_WIDTH/2), 0, Math.PI*2);
    ctx.fill();
    ctx.restore();
    strokeDrawn = true;
  }
  if (!hasInk(canvas)) return;
  if (submitTimer) clearTimeout(submitTimer);
  submitTimer = setTimeout(() => { if (hasInk(canvas)) predictAndHandle({append:true, clearAfter:true}); submitTimer=null; }, submitDelayMs);
}

canvas.addEventListener('mousedown', startDraw);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', endDraw);
canvas.addEventListener('mouseleave', endDraw);
// Ensure simple click places a dot (no drag required)
canvas.addEventListener('click', (e)=>{
  // If a stroke was drawn in this interaction, skip (drag case)
  if (drawing) return;
  if (strokeDrawn) return;
  const {x,y}=getPos(e);
  ctx.save();
  ctx.beginPath();
  ctx.fillStyle = STROKE_COLOR;
  ctx.arc(x,y, Math.max(1, STROKE_WIDTH/2), 0, Math.PI*2);
  ctx.fill();
  ctx.restore();
  strokeDrawn = true;
  if (submitTimer) clearTimeout(submitTimer);
  submitTimer = setTimeout(()=>{ if(hasInk(canvas)) predictAndHandle({append:true, clearAfter:true}); submitTimer=null; }, submitDelayMs);
});
canvas.addEventListener('touchstart', (e)=>{e.preventDefault();startDraw(e);});
canvas.addEventListener('touchmove', (e)=>{e.preventDefault();draw(e);});
canvas.addEventListener('touchend', (e)=>{e.preventDefault();endDraw(e);});
canvas.addEventListener('contextmenu', async (e)=>{e.preventDefault(); if (submitTimer){clearTimeout(submitTimer); submitTimer=null;} if (hasInk(canvas)) await predictAndHandle({append:true, clearAfter:true});});

// Model
async function ensureModel(){ if(!model){ model = await tf.loadLayersModel('model/model.json'); } }

async function predictAndHandle({append, clearAfter}){
  try{
    await ensureModel();
    const input = tf.tidy(()=> preprocessCanvasTo28x28(canvas).expandDims(0));
    const logits = model.predict(input);
    const probs = logits.dataSync();
    input.dispose(); tf.dispose(logits);
    let idx = argmaxIndex(probs);
    idx = adjustAmbiguitiesLetters(idx, probs, canvas);
    showResult(probs, idx);
    if (append){ const tgt=currentTarget(); if (tgt) recordAttempt(tgt, probs, idx); handleGuess(LABELS[idx]); }
    if (clearAfter){ clearCanvas(); strokeDrawn=false; }
  }catch(err){ console.error(err); alert('Prediction failed. Ensure /model/model.json exists and you are serving from repo root.'); }
}

function argmaxIndex(arr){ let idx=0,val=arr[0]; for(let i=1;i<arr.length;i++){ if(arr[i]>val){val=arr[i]; idx=i;} } return idx; }

function clearCanvas(){ ctx.clearRect(0,0,CANVAS_SIZE,CANVAS_SIZE); ctx.fillStyle='#000'; ctx.fillRect(0,0,CANVAS_SIZE,CANVAS_SIZE); ctx.fillStyle='#000'; }
function hasInk(cnv){ const w=cnv.width,h=cnv.height; const data=cnv.getContext('2d').getImageData(0,0,w,h).data; const th=10; for(let i=0;i<data.length;i+=4){ const v=(data[i]+data[i+1]+data[i+2])/3; if(v>th) return true; } return false; }

function preprocessCanvasTo28x28(srcCanvas){
  const srcCtx = srcCanvas.getContext('2d');
  const img = srcCtx.getImageData(0,0,CANVAS_SIZE,CANVAS_SIZE);
  let minX=CANVAS_SIZE,minY=CANVAS_SIZE,maxX=0,maxY=0; const th=10;
  for(let y=0;y<CANVAS_SIZE;y++){ for(let x=0;x<CANVAS_SIZE;x++){ const i=(y*CANVAS_SIZE+x)*4; const v=(img.data[i]+img.data[i+1]+img.data[i+2])/3; if(v>th){ if(x<minX)minX=x; if(y<minY)minY=y; if(x>maxX)maxX=x; if(y>maxY)maxY=y; } } }
  if(minX>maxX||minY>maxY){ minX=0;minY=0;maxX=CANVAS_SIZE-1;maxY=CANVAS_SIZE-1; }
  const bw=maxX-minX+1, bh=maxY-minY+1, boxSize=Math.max(bw,bh);
  const tmp=document.createElement('canvas'); tmp.width=boxSize; tmp.height=boxSize; const c=tmp.getContext('2d'); c.fillStyle='#000'; c.fillRect(0,0,boxSize,boxSize); const dx=(boxSize-bw)/2, dy=(boxSize-bh)/2; c.drawImage(srcCanvas,minX,minY,bw,bh,dx,dy,bw,bh);
  const target=document.createElement('canvas'); target.width=28; target.height=28; const t=target.getContext('2d'); t.fillStyle='#000'; t.fillRect(0,0,28,28);
  const scale=20/boxSize; const w=Math.max(1,Math.round(boxSize*scale)); const h=Math.max(1,Math.round(boxSize*scale)); const sx=Math.floor((28-w)/2); const sy=Math.floor((28-h)/2); t.imageSmoothingEnabled=true; t.imageSmoothingQuality='high'; t.drawImage(tmp,0,0,boxSize,boxSize,sx,sy,w,h);
  const img2=t.getImageData(0,0,28,28); const buf=new Float32Array(28*28);
  for(let y=0;y<28;y++){ for(let x=0;x<28;x++){ const srcX = FLIP_INPUT_HORIZONTAL ? (27-x) : x; const idx=(y*28+srcX)*4; const r=img2.data[idx], g=img2.data[idx+1], b=img2.data[idx+2]; buf[y*28+x]=(r+g+b)/(3*255); } }
  return tf.tensor(buf,[28,28,1]);
}

function showResult(probs, topIdxOverride=null){
  const rows=[]; for(let d=0; d<probs.length; d++) rows.push({d,p:probs[d]}); rows.sort((a,b)=>b.p-a.p);
  const top = rows[0]; const topIdx = (topIdxOverride!=null)? topIdxOverride : top.d; const predEl=document.getElementById('predDigit'); if (predEl) predEl.textContent = LABELS[topIdx];
  const probsEl=document.getElementById('probs'); if (probsEl){ probsEl.innerHTML=''; for(const {d,p} of rows){ const pct=Math.round(p*1000)/10; if(pct<=0) continue; const row=document.createElement('div'); row.className='bar'; row.innerHTML = `<div>${LABELS[d]}</div><div class="track"><div class="fill" style="width:${Math.min(100,pct)}%"></div></div><div class="pct">${pct.toFixed(1)}%</div>`; probsEl.appendChild(row);} }
}

function getInkBoundingBox(srcCanvas){ const ctx2=srcCanvas.getContext('2d'); const img=ctx2.getImageData(0,0,CANVAS_SIZE,CANVAS_SIZE); let minX=CANVAS_SIZE,minY=CANVAS_SIZE,maxX=-1,maxY=-1; const th=10; for(let y=0;y<CANVAS_SIZE;y++){ for(let x=0;x<CANVAS_SIZE;x++){ const i=(y*CANVAS_SIZE+x)*4; const v=(img.data[i]+img.data[i+1]+img.data[i+2])/3; if(v>th){ if(x<minX)minX=x; if(y<minY)minY=y; if(x>maxX)maxX=x; if(y>maxY)maxY=y; } } } if(maxX<0||maxY<0) return null; return {x:minX,y:minY,w:maxX-minX+1,h:maxY-minY+1}; }
function leftColumnCoverage(srcCanvas, frac=0.15){ const ctx2=srcCanvas.getContext('2d'); const w=CANVAS_SIZE,h=CANVAS_SIZE; const img=ctx2.getImageData(0,0,w,h); const colLimit=Math.floor(w*frac); const th=10; let rows=0; for(let y=0;y<h;y++){ for(let x=0;x<colLimit;x++){ const i=(y*w+x)*4; const v=(img.data[i]+img.data[i+1]+img.data[i+2])/3; if(v>th){ rows++; break; } } } return rows/h; }
function rightColumnCoverage(srcCanvas, frac=0.15){ const ctx2=srcCanvas.getContext('2d'); const w=CANVAS_SIZE,h=CANVAS_SIZE; const img=ctx2.getImageData(0,0,w,h); const start=Math.floor(w*(1-frac)); const th=10; let rows=0; for(let y=0;y<h;y++){ for(let x=start;x<w;x++){ const i=(y*w+x)*4; const v=(img.data[i]+img.data[i+1]+img.data[i+2])/3; if(v>th){ rows++; break; } } } return rows/h; }
function getTopKIndices(probs,k){ const idxs=probs.map((p,i)=>i).sort((a,b)=>probs[b]-probs[a]); return idxs.slice(0,k); }
function adjustAmbiguitiesLetters(idx, probs, srcCanvas){ const top2=getTopKIndices(probs,2); const labelAt=(i)=>LABELS[i]; const setHas=(arr,ch)=>arr.some((i)=>labelAt(i)===ch); if(setHas(top2,'b')&&setHas(top2,'d')){ const left=leftColumnCoverage(srcCanvas,0.12); const right=rightColumnCoverage(srcCanvas,0.12); if(left>right+0.15) return LABELS.indexOf('b'); if(right>left+0.15) return LABELS.indexOf('d'); } if(setHas(top2,'p')&&setHas(top2,'q')){ const left=leftColumnCoverage(srcCanvas,0.12); const right=rightColumnCoverage(srcCanvas,0.12); if(left>right+0.15) return LABELS.indexOf('p'); if(right>left+0.15) return LABELS.indexOf('q'); } return idx; }

// Game helpers
function currentTarget(){ return (game.index < game.target.length) ? game.target[game.index] : null; }
function randLetter(){ return LABELS[Math.floor(Math.random()*LABELS.length)]; }
function makeTarget(n=20){ let s=''; for(let i=0;i<n;i++) s+=randLetter(); return s; }
function buildTape(){ const tape=document.getElementById('tape'); if(!tape) return; tape.innerHTML=''; for(let i=0;i<game.target.length;i++){ const el=document.createElement('div'); el.className='cell'; el.textContent=game.target[i]; tape.appendChild(el);} updateTape(); updateCurrentLetter(); }
function updateTape(){ const tape=document.getElementById('tape'); if(!tape) return; const children=tape.children; for(let i=0;i<children.length;i++){ const el=children[i]; el.classList.remove('done','current','wrong'); if(i<game.index) el.classList.add('done'); else if(i===game.index) el.classList.add(game.lastWrong?'wrong':'current'); } const current=children[game.index]; const offset=current ? current.offsetLeft : 0; tape.style.transform=`translateX(${-offset}px)`; updateCurrentLetter(); updateStats(); }
function updateCurrentLetter(){ const el=document.getElementById('currentLetter'); if(!el) return; if(game.index>=game.target.length){ el.textContent='✓'; } else { el.textContent = game.target[game.index] || '–'; } }
function startClock(){ game.started=true; game.startMs=Date.now(); game.endMs=0; if(game.timerId) clearInterval(game.timerId); game.timerId=setInterval(updateStats,100); }
function finishClock(){ game.endMs=Date.now(); if(game.timerId){clearInterval(game.timerId); game.timerId=null;} updateStats(); showDiagnostics(); }
function elapsedMs(){ if(!game.started) return 0; return (game.endMs||Date.now())-game.startMs; }
function updateStats(){ const elElapsed=document.getElementById('elapsed'); const elLpm=document.getElementById('lpm'); const ms=elapsedMs(); if(elElapsed) elElapsed.textContent=(ms/1000).toFixed(1)+'s'; const lettersDone=Math.min(game.index, game.target.length); const minutes=Math.max(ms/60000,1e-6); const lpm=lettersDone/minutes; if(elLpm) elLpm.textContent=lpm.toFixed(1); }
function handleGuess(ch){ if(!game.started){ startClock(); const hint=document.getElementById('canvasHint'); if(hint) hint.style.display='none'; } if(game.index >= game.target.length) return; const want=game.target[game.index]; if(ch===want){ game.index+=1; game.lastWrong=false; if(game.index>=game.target.length) finishClock(); } else { game.lastWrong=true; } updateTape(); }
function resetGame(){ game.target=makeTarget(20); game.index=0; game.started=false; game.startMs=0; game.endMs=0; game.lastWrong=false; game.logs=[]; if(game.timerId){clearInterval(game.timerId); game.timerId=null;} buildTape(); clearCanvas(); const probsEl=document.getElementById('probs'); if(probsEl) probsEl.innerHTML=''; const predEl=document.getElementById('predDigit'); if(predEl) predEl.textContent='–'; const elElapsed=document.getElementById('elapsed'); if(elElapsed) elElapsed.textContent='0.0s'; const elLpm=document.getElementById('lpm'); if(elLpm) elLpm.textContent='0.0'; const hint=document.getElementById('canvasHint'); if(hint) hint.style.display='flex'; updateCurrentLetter(); }
const resetBtn=document.getElementById('resetGame'); if(resetBtn) resetBtn.addEventListener('click', resetGame); resetGame();

// Delay slider
const delaySlider=document.getElementById('delaySlider'); const delayValue=document.getElementById('delayValue');
if(delaySlider && delayValue){ const setDelay=(v)=>{ const n=Math.max(0,Math.min(1000,parseInt(v,10)||0)); submitDelayMs=n; delayValue.textContent=String(n); }; setDelay(delaySlider.value||'300'); delaySlider.addEventListener('input',(e)=>{ setDelay(e.target.value); if(submitTimer){ clearTimeout(submitTimer); submitTimer=setTimeout(()=>{ if(hasInk(canvas)) predictAndHandle({append:true, clearAfter:true}); submitTimer=null; }, submitDelayMs); } }); }

// Logging + diagnostics
function recordAttempt(targetChar, probs, predIdx){ const correctIdx=LABELS.indexOf(targetChar); let pCorrect=0,pOtherMax=0; for(let i=0;i<probs.length;i++){ const p=probs[i]; if(i===correctIdx) pCorrect=p; else if(p>pOtherMax) pOtherMax=p; } const correct=(predIdx===correctIdx); game.logs.push({target:targetChar, predIdx, pCorrect, pOtherMax, correct}); }
function showDiagnostics(){
  const modal=document.getElementById('diagModal'); if(!modal) return;
  // Aggregate attempts per letter
  const per=new Map();
  for (const log of game.logs){
    const e=per.get(log.target)||{attempts:0,wrong:0,sumConf:0,sumMargin:0};
    e.attempts += 1;
    if (!log.correct) e.wrong += 1;
    e.sumConf += log.pCorrect;
    e.sumMargin += (log.pCorrect - log.pOtherMax);
    per.set(log.target, e);
  }

  // Build full table rows: Letter | Accuracy% | Avg Conf% | Avg Gap%
  const tbody=document.getElementById('diagTable'); if (tbody) tbody.innerHTML='';
  const letters=Array.from(per.keys()).sort();
  for (const ch of letters){
    const e=per.get(ch);
    const att=e.attempts;
    const accPct = att ? (100*(att - e.wrong)/att) : 0;
    const avgConf = att ? (100*e.sumConf/att) : 0;
    const avgMargin = att ? (100*e.sumMargin/att) : 0;
    if (tbody){
      const tr=document.createElement('tr');
      tr.innerHTML = `<td>${ch}</td><td>${accPct.toFixed(1)}%</td><td>${avgConf.toFixed(1)}%</td><td>${avgMargin.toFixed(1)}%</td>`;
      tbody.appendChild(tr);
    }
  }

  // Big metrics
  const totalAttempts = game.logs.length;
  const correctAttempts = game.logs.reduce((a,l)=> a + (l.correct?1:0), 0);
  const overallAcc = totalAttempts ? (100*correctAttempts/totalAttempts) : 0;
  const overallAvgConf = totalAttempts ? (100*game.logs.reduce((a,l)=>a+l.pCorrect,0)/totalAttempts) : 0;
  const lpmText = (document.getElementById('lpm')?.textContent || '0.0');
  const diagLpmBig=document.getElementById('diagLpmBig'); if(diagLpmBig) diagLpmBig.textContent = lpmText;
  const diagAcc=document.getElementById('diagAcc'); if(diagAcc) diagAcc.textContent = overallAcc.toFixed(1) + '%';
  const diagAvgConf=document.getElementById('diagAvgConf'); if(diagAvgConf) diagAvgConf.textContent = overallAvgConf.toFixed(1) + '%';
  const diagAttempts=document.getElementById('diagAttempts'); if(diagAttempts) diagAttempts.textContent=String(totalAttempts);

  // Top and bottom 3 by avg gap (clarity)
  const rowsForRank = letters.map(ch => {
    const e=per.get(ch);
    const att=e.attempts;
    return {
      ch,
      att,
      accPct: att ? (100*(att - e.wrong)/att) : 0,
      avgConf: att ? (100*e.sumConf/att) : 0,
      avgMargin: att ? (100*e.sumMargin/att) : 0,
    };
  }).filter(r=>r.att>0);
  rowsForRank.sort((a,b)=> b.avgMargin - a.avgMargin);
  const top3 = rowsForRank.slice(0,3);
  const bottom3 = rowsForRank.slice(-3).reverse();

  const topBody = document.getElementById('diagTopTable');
  if (topBody) {
    topBody.innerHTML='';
    for (const r of top3){
      const tr=document.createElement('tr');
      tr.innerHTML = `<td>${r.ch}</td><td>${r.accPct.toFixed(1)}%</td><td>${r.avgConf.toFixed(1)}%</td><td>${r.avgMargin.toFixed(1)}%</td>`;
      topBody.appendChild(tr);
    }
  }
  const bottomBody = document.getElementById('diagBottomTable');
  if (bottomBody) {
    bottomBody.innerHTML='';
    for (const r of bottom3){
      const tr=document.createElement('tr');
      tr.innerHTML = `<td>${r.ch}</td><td>${r.accPct.toFixed(1)}%</td><td>${r.avgConf.toFixed(1)}%</td><td>${r.avgMargin.toFixed(1)}%</td>`;
      bottomBody.appendChild(tr);
    }
  }

  modal.hidden=false;
}
const diagClose=document.getElementById('diagClose'); if(diagClose) diagClose.addEventListener('click', ()=>{ const m=document.getElementById('diagModal'); if(m) m.hidden=true; });
