// Browser ports of the repo's Python code, written so the page can rerun it live.
// Learn: level-1-learn/clip_numpy.py and the train/recall grouping of run_learn.py.
// Build: question_type, answer_question, scoring and token counts from level-2-build/qa_service.py.
// Deep: make_documents, text_layer, gold_answer, make_cases, text_answer from level-3-deep/deep_eval.py.
// NumPy's PCG64 generator cannot be reproduced here, so random draws use a seeded mulberry32:
// live numbers come from the same procedure with a different RNG, never from the recorded run.

export const IMAGE_SIZE = 16, EMBED_DIM = 32, HIDDEN_DIM = 128, TEXT_DIM = 24, TEMPERATURE = 0.1, PIX = 768;
export const COLOURS = ['red', 'green', 'blue', 'yellow'];
export const COLOUR_RGB = { red: [0.9, 0.15, 0.15], green: [0.15, 0.8, 0.25], blue: [0.2, 0.35, 0.95], yellow: [0.9, 0.8, 0.15] };
export const SHAPES = ['circle', 'square', 'triangle'];
export const VOCAB = ['<pad>', 'a', 'red', 'green', 'blue', 'yellow', 'circle', 'square', 'triangle'];
export const TOKEN = Object.fromEntries(VOCAB.map((w, i) => [w, i]));
export const BACKGROUND = 0.05;
export const COMBOS = COLOURS.flatMap((c) => SHAPES.map((s) => `${c} ${s}`));

// ---- randomness ----
export function mulberry32(seed) {
  return () => { seed |= 0; seed = (seed + 0x6d2b79f5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const randInt = (rand, lo, hi) => lo + Math.floor(rand() * (hi - lo)); // like rng.integers(lo, hi): hi exclusive
function normal(rand) { let u = 0; while (u === 0) u = rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand()); }

// ---- rendering (clip_numpy.render_image, deep_eval._draw) ----
function inShape(shape, x, y, cx, cy, size) {
  if (shape === 'circle') return (x - cx) ** 2 + (y - cy) ** 2 <= size ** 2;
  if (shape === 'square') return Math.abs(x - cx) <= size && Math.abs(y - cy) <= size;
  if (shape === 'triangle') return y >= cy - size && y <= cy + size && Math.abs(x - cx) <= (y - (cy - size)) / 2;
  throw new Error('unknown shape: ' + shape);
}
export function blankImage() { return new Float64Array(PIX).fill(BACKGROUND); }
export function draw(img, colour, shape, cx, cy, size) {
  const rgb = COLOUR_RGB[colour];
  for (let y = 0; y < IMAGE_SIZE; y++) for (let x = 0; x < IMAGE_SIZE; x++)
    if (inShape(shape, x, y, cx, cy, size)) { const o = (y * IMAGE_SIZE + x) * 3; img[o] = rgb[0]; img[o + 1] = rgb[1]; img[o + 2] = rgb[2]; }
  return img;
}
// Offsets dx, dy in -3..3 and size 4..5, as render_image draws them.
export function renderAt(colour, shape, dx, dy, size) { return draw(blankImage(), colour, shape, 7.5 + dx, 7.5 + dy, size); }
export function renderImage(colour, shape, rand) {
  const dx = randInt(rand, -3, 4), dy = randInt(rand, -3, 4), size = randInt(rand, 4, 6);
  return renderAt(colour, shape, dx, dy, size);
}
export const captionTokens = (colour, shape) => [TOKEN.a, TOKEN[colour], TOKEN[shape]];

export function makePairs(nPerCombo, seedOffset) {
  const n = 12 * nPerCombo, images = new Float64Array(n * PIX), tokens = [], combos = [];
  let idx = 0;
  COLOURS.forEach((colour, ci) => SHAPES.forEach((shape, si) => {
    for (let k = 0; k < nPerCombo; k++) {
      images.set(renderImage(colour, shape, mulberry32(seedOffset + idx)), idx * PIX);
      tokens.push(captionTokens(colour, shape)); combos.push(ci * 3 + si); idx++;
    }
  }));
  return { images, tokens, combos, n, nPerCombo };
}

// ---- model (row-major Float64Array matrices) ----
export function initParams(seed = 42) {
  const r = mulberry32(seed);
  const nrm = (len, sd) => Float64Array.from({ length: len }, () => normal(r) * sd);
  return {
    W1: nrm(PIX * HIDDEN_DIM, Math.sqrt(2 / PIX)), b1: new Float64Array(HIDDEN_DIM),
    W2: nrm(HIDDEN_DIM * EMBED_DIM, Math.sqrt(2 / HIDDEN_DIM)), b2: new Float64Array(EMBED_DIM),
    E: nrm(VOCAB.length * TEXT_DIM, 0.5),
    Wt: nrm(TEXT_DIM * EMBED_DIM, Math.sqrt(2 / TEXT_DIM)), bt: new Float64Array(EMBED_DIM),
  };
}
// out[n x k] = A[n x m] @ B[m x k] + bias
function matmul(A, B, n, m, k, bias) {
  const out = new Float64Array(n * k);
  for (let i = 0; i < n; i++) {
    const o = i * k;
    if (bias) out.set(bias, o);
    for (let j = 0; j < m; j++) { const a = A[i * m + j]; if (a === 0) continue; const b = j * k; for (let c = 0; c < k; c++) out[o + c] += a * B[b + c]; }
  }
  return out;
}
function normaliseRows(Z, n, d) {
  const norms = new Float64Array(n), out = new Float64Array(n * d);
  for (let i = 0; i < n; i++) { let s = 0; for (let c = 0; c < d; c++) s += Z[i * d + c] ** 2; norms[i] = Math.sqrt(s); for (let c = 0; c < d; c++) out[i * d + c] = Z[i * d + c] / norms[i]; }
  return { z: out, norms };
}
function imageForward(P, X, n) {
  const hpre = matmul(X, P.W1, n, PIX, HIDDEN_DIM, P.b1), h = hpre.map((v) => (v > 0 ? v : 0));
  const pre = matmul(h, P.W2, n, HIDDEN_DIM, EMBED_DIM, P.b2);
  return { hpre, h, ...normaliseRows(pre, n, EMBED_DIM) };
}
function textForward(P, tokens) {
  const n = tokens.length, pooled = new Float64Array(n * TEXT_DIM), counts = [];
  tokens.forEach((t, i) => {
    const live = t.filter((w) => w !== TOKEN['<pad>']); counts.push(live.length);
    for (const w of live) for (let c = 0; c < TEXT_DIM; c++) pooled[i * TEXT_DIM + c] += P.E[w * TEXT_DIM + c] / live.length;
  });
  const pre = matmul(pooled, P.Wt, n, TEXT_DIM, EMBED_DIM, P.bt);
  return { pooled, counts, ...normaliseRows(pre, n, EMBED_DIM) };
}
export const encodeImage = (P, X, n) => imageForward(P, X, n).z;
export const encodeText = (P, tokens) => textForward(P, tokens).z;

function softmaxes(I, T, n, tau) {
  const L = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { let s = 0; for (let c = 0; c < EMBED_DIM; c++) s += I[i * EMBED_DIM + c] * T[j * EMBED_DIM + c]; L[i * n + j] = s / tau; }
  const row = new Float64Array(n * n), col = new Float64Array(n * n);
  for (let i = 0; i < n; i++) { let m = -Infinity, s = 0; for (let j = 0; j < n; j++) m = Math.max(m, L[i * n + j]); for (let j = 0; j < n; j++) s += (row[i * n + j] = Math.exp(L[i * n + j] - m)); for (let j = 0; j < n; j++) row[i * n + j] /= s; }
  for (let j = 0; j < n; j++) { let m = -Infinity, s = 0; for (let i = 0; i < n; i++) m = Math.max(m, L[i * n + j]); for (let i = 0; i < n; i++) s += (col[i * n + j] = Math.exp(L[i * n + j] - m)); for (let i = 0; i < n; i++) col[i * n + j] /= s; }
  return { row, col };
}
// Two-direction InfoNCE over matched pairs (clip_numpy.contrastive_loss). I, T: flat n x d, unit rows.
export function contrastiveLoss(I, T, n, tau = TEMPERATURE, d = EMBED_DIM) {
  const dot = (a, i, b, j) => { let s = 0; for (let c = 0; c < d; c++) s += a[i * d + c] * b[j * d + c]; return s / tau; };
  const ce = (A, B) => { let tot = 0; for (let i = 0; i < n; i++) { const l = Array.from({ length: n }, (_, j) => dot(A, i, B, j)); const m = Math.max(...l); const lse = Math.log(l.reduce((s, v) => s + Math.exp(v - m), 0)) + m; tot += lse - l[i]; } return tot / n; };
  return 0.5 * (ce(I, T) + ce(T, I));
}

export function lossAndGrads(P, X, tokens, tau = TEMPERATURE) {
  const n = tokens.length, D = EMBED_DIM;
  const img = imageForward(P, X, n), txt = textForward(P, tokens);
  const { row, col } = softmaxes(img.z, txt.z, n, tau);
  let loss = 0;
  for (let i = 0; i < n; i++) loss -= Math.log(row[i * n + i]) + Math.log(col[i * n + i]);
  loss *= 0.5 / n;
  const dL = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) dL[i * n + j] = 0.5 * (row[i * n + j] + col[i * n + j] - 2 * (i === j)) / n;
  const dI = new Float64Array(n * D), dT = new Float64Array(n * D);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { const g = dL[i * n + j] / tau; for (let c = 0; c < D; c++) { dI[i * D + c] += g * txt.z[j * D + c]; dT[j * D + c] += g * img.z[i * D + c]; } }
  const backNorm = (dz, z, norms) => { const out = new Float64Array(n * D); for (let i = 0; i < n; i++) { let s = 0; for (let c = 0; c < D; c++) s += dz[i * D + c] * z[i * D + c]; for (let c = 0; c < D; c++) out[i * D + c] = (dz[i * D + c] - z[i * D + c] * s) / norms[i]; } return out; };
  const dIp = backNorm(dI, img.z, img.norms), dTp = backNorm(dT, txt.z, txt.norms);
  const g = { W1: new Float64Array(P.W1.length), b1: new Float64Array(HIDDEN_DIM), W2: new Float64Array(P.W2.length), b2: new Float64Array(D), E: new Float64Array(P.E.length), Wt: new Float64Array(P.Wt.length), bt: new Float64Array(D) };
  const dh = new Float64Array(n * HIDDEN_DIM);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < D; c++) g.b2[c] += dIp[i * D + c];
    for (let h = 0; h < HIDDEN_DIM; h++) {
      const hv = img.h[i * HIDDEN_DIM + h]; let s = 0;
      for (let c = 0; c < D; c++) { g.W2[h * D + c] += hv * dIp[i * D + c]; s += dIp[i * D + c] * P.W2[h * D + c]; }
      dh[i * HIDDEN_DIM + h] = img.hpre[i * HIDDEN_DIM + h] > 0 ? s : 0;
    }
    for (let h = 0; h < HIDDEN_DIM; h++) g.b1[h] += dh[i * HIDDEN_DIM + h];
    for (let p = 0; p < PIX; p++) { const x = X[i * PIX + p]; const o = p * HIDDEN_DIM; for (let h = 0; h < HIDDEN_DIM; h++) g.W1[o + h] += x * dh[i * HIDDEN_DIM + h]; }
    const dPool = new Float64Array(TEXT_DIM);
    for (let c = 0; c < D; c++) g.bt[c] += dTp[i * D + c];
    for (let t = 0; t < TEXT_DIM; t++) { const pv = txt.pooled[i * TEXT_DIM + t]; for (let c = 0; c < D; c++) { g.Wt[t * D + c] += pv * dTp[i * D + c]; dPool[t] += dTp[i * D + c] * P.Wt[t * D + c]; } }
    for (const w of tokens[i]) if (w !== TOKEN['<pad>']) for (let t = 0; t < TEXT_DIM; t++) g.E[w * TEXT_DIM + t] += dPool[t] / txt.counts[i];
  }
  return { loss, grads: g };
}

export function makeAdam(P, lr = 0.01, b1 = 0.9, b2 = 0.999, eps = 1e-8) {
  const m = {}, v = {}; for (const k in P) { m[k] = new Float64Array(P[k].length); v[k] = new Float64Array(P[k].length); }
  let t = 0;
  return (grads) => {
    t++;
    const c1 = 1 - b1 ** t, c2 = 1 - b2 ** t;
    for (const k in P) { const p = P[k], g = grads[k], mk = m[k], vk = v[k]; for (let i = 0; i < p.length; i++) { mk[i] = b1 * mk[i] + (1 - b1) * g[i]; vk[i] = b2 * vk[i] + (1 - b2) * g[i] * g[i]; p[i] -= lr * (mk[i] / c1) / (Math.sqrt(vk[i] / c2) + eps); } }
  };
}

// Trainer with the train() batching of clip_numpy: one pair per distinct caption per step,
// cycling each caption group in order. step(k) advances k steps so the page can train in slices.
export function makeTrainer(data, seed = 42, lr = 0.01) {
  const P = initParams(seed), adam = makeAdam(P, lr), losses = [];
  const X = new Float64Array(12 * PIX);
  let steps = 0;
  return {
    params: P, losses,
    get steps() { return steps; },
    step(k = 1) {
      for (let s = 0; s < k; s++) {
        const idx = Array.from({ length: 12 }, (_, g) => g * data.nPerCombo + (steps % data.nPerCombo));
        idx.forEach((r, b) => X.set(data.images.subarray(r * PIX, (r + 1) * PIX), b * PIX));
        const { loss, grads } = lossAndGrads(P, X, idx.map((r) => data.tokens[r]));
        losses.push(loss); adam(grads); steps++;
      }
    },
  };
}

export function recallAtK(I, T, combos, k, direction, nI = combos.length, nT = combos.length) {
  const q = direction === 'image_to_text' ? I : T, c = direction === 'image_to_text' ? T : I;
  const nq = direction === 'image_to_text' ? nI : nT, nc = direction === 'image_to_text' ? nT : nI;
  let hits = 0;
  for (let i = 0; i < nq; i++) {
    const sims = Array.from({ length: nc }, (_, j) => { let s = 0; for (let d = 0; d < EMBED_DIM; d++) s += q[i * EMBED_DIM + d] * c[j * EMBED_DIM + d]; return [s, j]; });
    sims.sort((a, b) => b[0] - a[0]);
    if (sims.slice(0, k).some(([, j]) => combos[j] === combos[i])) hits++;
  }
  return hits / nq;
}
export function evaluate(P, data) {
  const I = encodeImage(P, data.images, data.n), T = encodeText(P, data.tokens);
  return {
    image_to_text_recall_at_1: recallAtK(I, T, data.combos, 1, 'image_to_text'),
    image_to_text_recall_at_5: recallAtK(I, T, data.combos, 5, 'image_to_text'),
    text_to_image_recall_at_1: recallAtK(I, T, data.combos, 1, 'text_to_image'),
    text_to_image_recall_at_5: recallAtK(I, T, data.combos, 5, 'text_to_image'),
  };
}

// ---- Build: qa_service.py ----
export const PATCH_SIZE = 4;
export const SOURCE = 'learn_encoder_stand_in';
export function visualTokenCount(imageSize = IMAGE_SIZE, patchSize = PATCH_SIZE) {
  const perSide = Math.floor(imageSize / patchSize), patches = perSide * perSide;
  return { patches, with_class_token: patches + 1 };
}
export const textTokenCount = (text) => text.split(/\s+/).filter(Boolean).length;
export const unifiedTokenCount = (q, imageSize, patchSize) => visualTokenCount(imageSize, patchSize).with_class_token + textTokenCount(q);

export function captionSims(P, img) {
  const T = encodeText(P, COMBOS.map((c) => { const [a, b] = c.split(' '); return captionTokens(a, b); }));
  const I = encodeImage(P, img, 1);
  return COMBOS.map((_, j) => { let s = 0; for (let d = 0; d < EMBED_DIM; d++) s += I[d] * T[j * EMBED_DIM + d]; return s; });
}
export function predictCombo(P, img) {
  const sims = captionSims(P, img);
  let best = 0; sims.forEach((s, j) => { if (s > sims[best]) best = j; });
  return { colour: COLOURS[Math.floor(best / 3)], shape: SHAPES[best % 3], confidence: Math.min(1, Math.max(0, sims[best])), sims };
}
export function questionType(question) {
  const q = question.toLowerCase();
  if (q.includes('how many')) return 'count';
  if (q.includes('colour') && q.includes('shape')) return 'both';
  if (q.includes('color') || q.includes('colour')) return 'colour';
  if (q.includes('shape')) return 'shape';
  return 'unsupported';
}
export function answerQuestion(P, img, question, pred = predictCombo(P, img)) {
  const { colour, shape, confidence } = pred, qt = questionType(question);
  if (qt === 'colour') return { answer: colour, colour, shape, count: 1, confidence, source: SOURCE };
  if (qt === 'shape') return { answer: shape, colour, shape, count: 1, confidence, source: SOURCE };
  if (qt === 'both') return { answer: `${colour} ${shape}`, colour, shape, count: 1, confidence, source: SOURCE };
  if (qt === 'count') return { answer: 'one', colour, shape, count: 1, confidence, source: SOURCE }; // constant, as in qa_service.py
  return { answer: 'unsupported question', colour: 'unknown', shape: 'unknown', count: 0, confidence: 0, source: SOURCE };
}
export const normalise = (t) => t.toLowerCase().split(/\s+/).filter(Boolean).join(' ');
export const exactMatch = (p, g) => (normalise(p) === normalise(g) ? 1 : 0);
export function tokenF1(p, g) {
  const count = (t) => normalise(t).split(' ').filter(Boolean).reduce((m, w) => m.set(w, (m.get(w) || 0) + 1), new Map());
  const P = count(p), G = count(g);
  if (!P.size || !G.size) return 0;
  let overlap = 0; for (const [w, c] of P) overlap += Math.min(c, G.get(w) || 0);
  if (!overlap) return 0;
  const np = [...P.values()].reduce((a, b) => a + b), ng = [...G.values()].reduce((a, b) => a + b);
  const pr = overlap / np, rc = overlap / ng;
  return (2 * pr * rc) / (pr + rc);
}
export const BUILD_QUESTIONS = { colour: 'what is the main colour', shape: 'what shape is in the image', both: 'what colour and shape is in the image', count: 'how many shapes are in the image' };

// ---- Deep: deep_eval.py ----
export const DOC_SEED_OFFSET = 700000;
export const NUM_WORDS = { 1: 'one', 2: 'two', 3: 'three', 4: 'four' };
export const TITLES = ['alpha', 'beta', 'gamma', 'delta'];
export const QUESTION_TYPES = ['colour', 'shape', 'both', 'count', 'title', 'left_shape'];
export const QUESTIONS = { ...BUILD_QUESTIONS, title: 'what is the document title', left_shape: 'what shape is on the left side' };
export const SLOTS = { 2: [[3, 7], [11, 7]], 3: [[2, 7], [7, 7], [12, 7]], 4: [[3, 3], [11, 3], [3, 11], [11, 11]] };
export const SLOT_SIZE = 2;
export const combo = (i) => { i %= 12; return [COLOURS[Math.floor(i / 3)], SHAPES[i % 3]]; };
export function renderDocument(items, docIndex) {
  if (items.length === 1) return renderImage(items[0][0], items[0][1], mulberry32(DOC_SEED_OFFSET + docIndex));
  const img = blankImage();
  items.forEach(([c, s], k) => draw(img, c, s, SLOTS[items.length][k][0], SLOTS[items.length][k][1], SLOT_SIZE));
  return img;
}
export const textLayer = (title, items) => `title ${title} figure shapes ${items.map(([, s]) => s).join(' ')}`;
export function makeDocuments() {
  return Array.from({ length: 24 }, (_, i) => {
    const n = [1, 2, 3, 4][i % 4], items = Array.from({ length: n }, (_, k) => combo(i + k)), title = TITLES[i % 4];
    return { doc_id: `d${String(i).padStart(2, '0')}`, item_count: n, items, first_colour: items[0][0], first_shape: items[0][1], title, text: textLayer(title, items), index: i };
  });
}
export function goldAnswer(doc, qt) {
  if (qt === 'colour') return doc.first_colour;
  if (qt === 'shape' || qt === 'left_shape') return doc.first_shape;
  if (qt === 'both') return `${doc.first_colour} ${doc.first_shape}`;
  if (qt === 'count') return NUM_WORDS[doc.item_count];
  if (qt === 'title') return doc.title;
  throw new Error('unknown question type: ' + qt);
}
export const makeCases = (docs) => docs.flatMap((doc) => QUESTION_TYPES.map((qt) => ({ case_id: `${doc.doc_id}-${qt}`, doc_id: doc.doc_id, item_count: doc.item_count, question_type: qt, question: QUESTIONS[qt], labelled_answer: goldAnswer(doc, qt), doc })));
const firstWord = (toks, vocab) => toks.find((t) => vocab.includes(t)) ?? 'unknown';
export function textAnswer(text, qt) {
  const toks = text.toLowerCase().split(/\s+/).filter(Boolean);
  if (qt === 'colour') return firstWord(toks, COLOURS);
  if (qt === 'shape' || qt === 'left_shape') return firstWord(toks, SHAPES);
  if (qt === 'both') return `${firstWord(toks, COLOURS)} ${firstWord(toks, SHAPES)}`;
  if (qt === 'count') { const i = toks.indexOf('shapes'); if (i < 0) return 'unknown'; return NUM_WORDS[toks.slice(i + 1).filter((t) => SHAPES.includes(t)).length] ?? 'unknown'; }
  if (qt === 'title') { const i = toks.indexOf('title'); return i >= 0 && i + 1 < toks.length ? toks[i + 1] : 'unknown'; }
  throw new Error('unknown question type: ' + qt);
}
// The whole 144-case study. params null skips the vision route (text route only).
export function runDeep(P) {
  const docs = makeDocuments(), preds = P ? docs.map((d) => predictCombo(P, renderDocument(d.items, d.index))) : null;
  return makeCases(docs).map((c) => {
    const t = textAnswer(c.doc.text, c.question_type), row = { case_id: c.case_id, doc_id: c.doc_id, item_count: c.item_count, question_type: c.question_type, labelled_answer: c.labelled_answer, text_answer: t, text_exact_match: exactMatch(t, c.labelled_answer), text_token_f1: tokenF1(t, c.labelled_answer) };
    if (P) { const v = answerQuestion(P, null, c.question, preds[c.doc.index]).answer; Object.assign(row, { vision_answer: v, vision_exact_match: exactMatch(v, c.labelled_answer), vision_token_f1: tokenF1(v, c.labelled_answer) }); }
    return row;
  });
}

// ---- aggregates and statistics over per-case rows ----
export const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
export function byType(rows, routes = ['vision', 'text']) {
  const out = {};
  for (const qt of QUESTION_TYPES) {
    const r = rows.filter((x) => x.question_type === qt); if (!r.length) continue;
    out[qt] = { cases: r.length };
    for (const k of routes) { out[qt][`${k}_exact_match`] = mean(r.map((x) => x[`${k}_exact_match`])); out[qt][`${k}_mean_token_f1`] = mean(r.map((x) => x[`${k}_token_f1`])); }
  }
  return out;
}
export function mcnemarExact(v, t) {
  let b = 0, c = 0; v.forEach((x, i) => { if (x === 1 && t[i] === 0) b++; if (x === 0 && t[i] === 1) c++; });
  const n = b + c, k = Math.min(b, c);
  let pmf = 0.5 ** n, cdf = 0;
  for (let i = 0; i <= k; i++) { cdf += pmf; pmf *= (n - i) / (i + 1); }
  return { vision_only_correct: b, text_only_correct: c, discordant: n, p_value: n === 0 ? 1 : Math.min(1, 2 * cdf) };
}
// numpy.percentile default (linear interpolation)
export function percentile(sorted, q) { const h = (sorted.length - 1) * q, lo = Math.floor(h); return sorted[lo] + (h - lo) * ((sorted[Math.min(lo + 1, sorted.length - 1)]) - sorted[lo]); }
// Percentile bootstrap of the mean of each array in `groups` (clusters of values), resampling clusters.
export function bootstrapCI(clusters, B = 2000, seed = 20261107) {
  const r = mulberry32(seed), n = clusters.length, size = clusters.reduce((s, c) => s + c.length, 0) / n, out = new Float64Array(B);
  const sums = clusters.map((c) => c.reduce((s, v) => s + v, 0));
  for (let b = 0; b < B; b++) { let s = 0; for (let i = 0; i < n; i++) s += sums[Math.floor(r() * n)]; out[b] = s / (n * size); }
  out.sort();
  return [percentile(out, 0.025), percentile(out, 0.975)];
}
