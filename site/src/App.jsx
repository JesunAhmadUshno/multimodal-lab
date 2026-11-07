import { useMemo, useState, useEffect, useCallback, useId, useRef } from 'react';
import learn from '../../level-1-learn/results_learn.json';
import build from '../../level-2-build/results_build.json';
import deep from '../../level-3-deep/results_deep.json';
import seeds from '../../analysis/results_seeds.json';
import stats from '../../analysis/results_stats.json';
import countBase from '../../analysis/results_count_baseline.json';
import anatomyMd from '../../level-1-learn/ANATOMY.md?raw';
import roadmapMd from '../../ROADMAP.md?raw';
import summaryMd from '../../docs/index.md?raw';
import qaSource from '../../level-2-build/qa_service.py?raw';
import * as L from './lab.js';

const REPO = 'https://github.com/JesunAhmadUshno/multimodal-lab';
const blob = (p) => `${REPO}/blob/main/${p}`;
const f3 = (x) => x.toFixed(3);
const pct = (x) => `${Math.round(x * 100)}%`;
const fmtP = (p) => (p < 1e-4 ? p.toExponential(1) : p.toFixed(4));

const SECTIONS = [['top', 'Overview', '◎'], ['learn', 'Trainer', '∇'], ['sketch', 'Sketchpad', '▦'], ['build', 'Build', '⌘'], ['study', 'Study', '▥'], ['counting', 'Counting', '#'], ['stats', 'Uncertainty', '±'], ['seeds', 'Seeds', '⟳'], ['anatomy', 'Anatomy', '◫'], ['limits', 'Limits', '!']];

// Text pulled from the repo's own documents at build time.
const grades = Object.fromEntries([...summaryMd.match(/Grades on record \(of 10,000\): (.*)\./)[1].matchAll(/(\w+) (\d{1,2},\d{3})/g)].map((m) => [m[1], m[2]]));
const anatomy = anatomyMd.split('\n## Part ').slice(1).map((block) => {
  const field = (name) => (block.match(new RegExp(`- ${name}: ([\\s\\S]*?)(?=\\n- |\\n\\n|$)`)) || [])[1]?.replace(/\s+/g, ' ');
  return { title: block.split('\n')[0], job: field('Job'), llava: field('In LLaVA'), fail: field('Failure tendency') };
});
const backlog = [...roadmapMd.split('## v1.1 backlog')[1].matchAll(/\n(\d)\. ([\s\S]*?)(?=\n\d\. |\n\n)/g)].map((m) => ({ n: m[1], text: m[2].replace(/\s+/g, ' ') }));
const countBranch = qaSource.split('\n').filter((l) => /qtype == "count"|answer, out_colour, out_shape, count = "one"/.test(l)).map((l) => l.trim()).join('\n');

function useReveal() {
  useEffect(() => {
    const els = document.querySelectorAll('.reveal');
    if (!('IntersectionObserver' in window)) { els.forEach((e) => e.classList.add('in')); return; }
    const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } }), { threshold: 0.08 });
    els.forEach((e) => io.observe(e));
    return () => io.disconnect();
  }, []);
}

function MenuBar() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => { const t = setInterval(() => setNow(new Date()), 30000); return () => clearInterval(t); }, []);
  return (
    <div className="menubar" role="banner">
      <span className="apple" aria-hidden="true">◆</span>
      <a className="app" href="#top">Multimodal Lab</a>
      <a className="m" href={blob('level-1-learn/README.md')}>Learn</a>
      <a className="m" href={blob('level-2-build/README.md')}>Build</a>
      <a className="m" href={blob('level-3-deep/README.md')}>Deep</a>
      <a className="m" href={blob('docs/research-review.md')}>Review</a>
      <span className="spacer" />
      <span className="status"><i /> CPU · NumPy · synthetic</span>
      <a href={REPO}>GitHub ↗</a>
      <span className="clock">{now.toLocaleString('en-US', { weekday: 'short', hour: 'numeric', minute: '2-digit' })}</span>
    </div>
  );
}

function Dock() {
  const [active, setActive] = useState('top');
  useEffect(() => {
    const io = new IntersectionObserver((es) => es.forEach((e) => e.isIntersecting && setActive(e.target.id)), { rootMargin: '-45% 0px -50% 0px' });
    SECTIONS.forEach(([id]) => { const el = document.getElementById(id); if (el) io.observe(el); });
    return () => io.disconnect();
  }, []);
  return (
    <nav className="dock" aria-label="Sections">
      {SECTIONS.map(([id, label, icon]) => (
        <a key={id} href={`#${id}`} className={active === id ? 'on' : ''} aria-current={active === id ? 'true' : undefined}>
          <span className="ico" aria-hidden="true">{icon}</span><span className="lbl">{label}</span>
        </a>
      ))}
    </nav>
  );
}

function Range({ label, value, min, max, step, onChange, show }) {
  const id = useId();
  return (
    <div className="field">
      <label htmlFor={id}>{label} <span>{show}</span></label>
      <input id={id} type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(+e.target.value)} />
    </div>
  );
}

function Select({ label, value, options, onChange }) {
  const id = useId();
  return (
    <div className="field"><label htmlFor={id}>{label}</label>
      <select id={id} value={value} onChange={(e) => onChange(e.target.value)}>{options.map((o) => <option key={o} value={o}>{o}</option>)}</select>
    </div>
  );
}

const Live = ({ children = 'live, your browser' }) => <span className="pill info">● {children}</span>;
const Recorded = ({ children = 'recorded run' }) => <span className="pill warn">◼ {children}</span>;

export default function App() {
  useReveal();
  const [model, setModel] = useState(null); // { params, seed, steps } once the trainer has run
  return (
    <div className="wrap">
      <MenuBar />
      <div className="glow" aria-hidden="true" />
      <Hero />
      <Trainer onTrained={setModel} />
      <Sketchpad model={model} />
      <Build />
      <Study model={model} />
      <Counting />
      <Uncertainty />
      <Seeds />
      <Anatomy />
      <Limits />
      <Dock />
      <footer>
        <a href={REPO}>Repository</a> · <a href={`${REPO}/releases`}>Releases</a> · <a href={`${REPO}/actions`}>CI</a> · Code Apache-2.0.
        <br />Recorded numbers are read from <code>level-*/results_*.json</code> and <code>analysis/results_*.json</code> at build time; everything marked live runs in your browser on synthetic data.
      </footer>
    </div>
  );
}

function Hero() {
  const countMulti = deep.vision_by_item_group.count_multi.vision_exact_match;
  return (
    <header className="card hero window" id="top" data-title="multimodal-lab - README">
      <div className="head">
        <p className="eyebrow">CLIP from scratch · QA service · document eval</p>
        <span className="pill warn">● synthetic data only</span>
      </div>
      <h1>A hand-built CLIP learns colour and shape from scratch, but on documents it beats a text parser only where the answer lives in the pixels, and it <em>cannot count</em>.</h1>
      <div className="banner" role="note">
        <b>Honest scope.</b> <code>data_is_public: {String(deep.data_is_public)}</code> <code>vlm_used: {String(deep.vlm_used)}</code>
        <span> {build.vlm_substitution.split(';')[0]}. The vision route is the Learn encoder plus a keyword composer, labelled <code>{L.SOURCE}</code>. No result here transfers to a real vision-language model.</span>
      </div>
      <p className="sub">
        Three levels, all NumPy on a CPU. Learn trains a CLIP-style dual encoder with hand-written backprop. Build wraps it in a schema-validated QA service. Deep pits that service against a text-only parser on {deep.eval_cases} synthetic document questions.
        Every window below either replays a recorded run (labelled) or reruns the code live in your browser.
      </p>
      <div className="actions">
        <a className="btn primary" href="#learn">Train the encoder live</a>
        <a className="btn ghost" href="#counting">See the counting failure</a>
      </div>
      <div className="tiles">
        <div className="tile"><div className="num">{f3(learn.before_training.image_to_text_recall_at_1)} → {f3(learn.after_training.image_to_text_recall_at_1)}</div><div className="cap">Learn image-to-text Recall@1 on {learn.held_out_pairs} held-out pairs</div></div>
        <div className="tile"><div className="num">{f3(build.exact_match)}</div><div className="cap">Build exact match on {build.eval_cases} cases: proves the loop, not VLM skill</div></div>
        <div className="tile"><div className="num">{f3(deep.overall.vision_exact_match)} <small>vs</small> {f3(deep.overall.text_exact_match)}</div><div className="cap">Deep exact match, vision route vs text-only parser</div></div>
        <div className="tile"><div className="num">{f3(countMulti)}</div><div className="cap">Vision count exact match on multi-item documents</div></div>
      </div>
    </header>
  );
}

// ---------- Learn: live trainer ----------
function LossChart({ losses }) {
  const W = 460, H = 220, pad = { l: 40, r: 10, t: 10, b: 24 };
  const maxStep = Math.max(learn.training_steps, losses.length);
  const x = (s) => pad.l + (s / maxStep) * (W - pad.l - pad.r);
  const y = (v) => pad.t + (1 - (Math.log10(Math.max(1e-3, Math.min(10, v))) + 3) / 4) * (H - pad.t - pad.b);
  const stride = Math.max(1, Math.floor(losses.length / 300));
  const pts = losses.filter((_, i) => i % stride === 0 || i === losses.length - 1).map((v, i, a) => `${x(i === a.length - 1 ? losses.length : i * stride)},${y(v)}`).join(' ');
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Training loss by step, live run against the recorded NumPy run">
      {[10, 1, 0.1, 0.01, 0.001].map((t) => <g key={t}><line className="axis" x1={pad.l} x2={W - pad.r} y1={y(t)} y2={y(t)} /><text x={pad.l - 6} y={y(t) + 4} textAnchor="end">{t}</text></g>)}
      {losses.length > 1 && <polyline fill="none" stroke="var(--primary)" strokeWidth="2" points={pts} />}
      <line x1={x(0)} x2={x(learn.training_steps)} y1={y(learn.train_loss_first)} y2={y(learn.train_loss_last)} stroke="var(--warn)" strokeDasharray="5 4" />
      {[[0, learn.train_loss_first], [learn.training_steps, learn.train_loss_last]].map(([s, v]) => <circle key={s} cx={x(s)} cy={y(v)} r="5" fill="var(--warn)" />)}
      <text x={x(learn.training_steps)} y={y(learn.train_loss_last) - 10} textAnchor="end" style={{ fill: 'var(--warn)' }}>recorded {learn.train_loss_last}</text>
      <text x={pad.l} y={H - 4}>step 0</text><text x={W - pad.r} y={H - 4} textAnchor="end">step {maxStep}</text>
    </svg>
  );
}

const RECALL_KEYS = [['image_to_text_recall_at_1', 'Image → text R@1'], ['image_to_text_recall_at_5', 'Image → text R@5'], ['text_to_image_recall_at_1', 'Text → image R@1'], ['text_to_image_recall_at_5', 'Text → image R@5']];

function Trainer({ onTrained }) {
  const [steps, setSteps] = useState(1000);
  const [seed, setSeed] = useState(learn.master_seed);
  const [busy, setBusy] = useState(false);
  const [state, setState] = useState(null);
  const data = useRef(null);
  const cancel = useRef(false);
  useEffect(() => () => { cancel.current = true; }, []);
  const run = useCallback(() => {
    setBusy(true);
    setState({ losses: [], done: 0, total: steps });
    setTimeout(() => {
      const t0 = performance.now();
      data.current ??= { train: L.makePairs(learn.train_pairs / 12, 100000), held: L.makePairs(learn.held_out_pairs / 12, 900000) };
      const before = L.evaluate(L.initParams(seed), data.current.held);
      const tr = L.makeTrainer(data.current.train, seed);
      const slice = () => {
        if (cancel.current) return;
        tr.step(Math.min(25, steps - tr.steps));
        if (tr.steps < steps) { setState({ losses: tr.losses.slice(), done: tr.steps, total: steps, before }); setTimeout(slice, 0); return; }
        const after = L.evaluate(tr.params, data.current.held);
        setState({ losses: tr.losses.slice(), done: tr.steps, total: steps, before, after, ms: performance.now() - t0, seed });
        onTrained({ params: tr.params, seed, steps });
        setBusy(false);
      };
      slice();
    }, 0);
  }, [steps, seed, onTrained]);
  const losses = state?.losses ?? [];
  return (
    <section id="learn" className="card window reveal" data-title="clip_numpy.py - Trainer">
      <div className="head">
        <div>
          <p className="eyebrow">Level 1 · Learn</p>
          <h2>Train the dual encoder in your browser: renderer, InfoNCE, backprop and Adam.</h2>
          <p className="sub">
            A line-for-line JavaScript port of <a href={blob('level-1-learn/clip_numpy.py')}>clip_numpy.py</a>: a {learn.image_shape.join('×')} image tower (one hidden layer of {L.HIDDEN_DIM}) and a bag-of-words text tower meet in a {learn.embed_dim}-dimensional space, trained with two-way contrastive loss at temperature {learn.temperature}, {learn.batch_pairs_per_step} pairs per step.
            NumPy's random generator cannot be reproduced in JavaScript, so this is a live retrain with its own RNG: expect numbers near, not equal to, the recorded run.
          </p>
        </div>
        <Live>live retrain in your browser, own RNG</Live>
      </div>
      <div className="controls">
        <Range label="Training steps" value={steps} min={100} max={learn.training_steps} step={100} onChange={setSteps} show={steps.toLocaleString()} />
        <div className="field"><label htmlFor="seed">Init seed</label><input id="seed" type="number" value={seed} onChange={(e) => setSeed(Math.trunc(+e.target.value) || 1)} /></div>
      </div>
      <div className="actions" style={{ marginTop: 0, marginBottom: 16 }}>
        <button className="btn primary" onClick={run} disabled={busy}>{busy ? `Training… ${state?.done ?? 0}/${steps}` : 'Run training'}</button>
        <span className="hint" aria-live="polite">{state?.after ? `Done: ${state.done} steps in ${(state.ms / 1000).toFixed(1)} s. Loss ${losses[0].toFixed(4)} → ${losses.at(-1).toFixed(6)}.` : busy ? 'Training in small slices so the page stays responsive.' : `Nothing runs until you press the button. ${learn.train_pairs.toLocaleString()} images are rendered first.`}</span>
      </div>
      <div className="track" style={{ marginBottom: 16 }} aria-hidden="true"><div className="fill" style={{ width: `${state ? (100 * state.done) / state.total : 0}%`, background: 'var(--primary)' }} /></div>
      <div className="grid2">
        <div>
          <h3>Training loss (log scale)</h3>
          <LossChart losses={losses} />
          <div className="legend"><span><i style={{ background: 'var(--primary)' }} />live, your browser</span><span><i style={{ background: 'var(--warn)' }} />recorded, NumPy, seed {learn.master_seed}: {learn.train_loss_first} → {learn.train_loss_last}</span></div>
        </div>
        <div>
          <h3>Held-out recall, before → after training</h3>
          <div className="bars" style={{ marginTop: 12 }}>
            {RECALL_KEYS.map(([k, label]) => (
              <div className="bar-row" key={k}>
                <div className="lab"><span>{label}</span><b>{state?.after ? `${f3(state.before[k])} → ${f3(state.after[k])}` : '-'}</b></div>
                <div className="track">
                  <div className="fill" style={{ width: `${(state?.after?.[k] ?? 0) * 100}%`, background: 'var(--primary)' }} />
                  <div className="ref" style={{ left: `${learn.after_training[k] * 100}%` }} title={`Recorded: ${learn.after_training[k]}`} />
                </div>
                <div className="hint">recorded, NumPy, seed {learn.master_seed}: {f3(learn.before_training[k])} → {f3(learn.after_training[k])}</div>
              </div>
            ))}
          </div>
          <p className="hint" style={{ marginTop: 12 }}>Black ticks mark the recorded after-training values on {learn.held_out_pairs} held-out pairs. Image → text R@5 always equals R@1 here: held-out captions repeat, so an image's top five captions are five copies of one caption.</p>
        </div>
      </div>
    </section>
  );
}

// ---------- Sketchpad ----------
const BRUSHES = [...L.COLOURS, 'erase'];
const rgbCss = (img, i) => `rgb(${Math.round(img[i * 3] * 255)},${Math.round(img[i * 3 + 1] * 255)},${Math.round(img[i * 3 + 2] * 255)})`;
const SKETCH_QS = ['colour', 'shape', 'both', 'count', 'title'];

function Sketchpad({ model }) {
  const [colour, setColour] = useState('red');
  const [shape, setShape] = useState('circle');
  const [dx, setDx] = useState(0);
  const [dy, setDy] = useState(0);
  const [size, setSize] = useState(4);
  const [items, setItems] = useState(1);
  const [brush, setBrush] = useState('blue');
  const [painted, setPainted] = useState(false);
  const [img, setImg] = useState(() => L.renderAt('red', 'circle', 0, 0, 4));
  const fallback = useMemo(() => L.initParams(learn.master_seed), []);
  const params = model?.params ?? fallback;
  useEffect(() => {
    const start = L.COMBOS.indexOf(`${colour} ${shape}`);
    setImg(items === 1 ? L.renderAt(colour, shape, dx, dy, size) : L.renderDocument(Array.from({ length: items }, (_, k) => L.combo(start + k)), 0));
    setPainted(false);
  }, [colour, shape, dx, dy, size, items]);
  const paint = (i) => {
    setImg((old) => { const n = old.slice(); const rgb = brush === 'erase' ? [L.BACKGROUND, L.BACKGROUND, L.BACKGROUND] : L.COLOUR_RGB[brush]; n.set(rgb, i * 3); return n; });
    setPainted(true);
  };
  const pred = useMemo(() => L.predictCombo(params, img), [params, img]);
  const W = 460, H = 250, padL = 104, bh = (H - 10) / 12;
  const span = Math.max(0.25, ...pred.sims.map(Math.abs)) * 1.05, lo = -span, hi = span;
  const sx = (v) => padL + ((v - lo) / (hi - lo)) * (W - padL - 30);
  const goldCount = painted ? null : items;
  return (
    <section id="sketch" className="card window reveal" data-title="retrieval.app - Sketchpad">
      <div className="head">
        <div>
          <p className="eyebrow">Retrieval and QA, live</p>
          <h2>Draw a figure and ask the service about it. Add shapes and watch the count stay at one.</h2>
          <p className="sub">
            The encoder scores your {L.IMAGE_SIZE}×{L.IMAGE_SIZE} image against all {L.COMBOS.length} captions, and the port of <a href={blob('level-2-build/qa_service.py')}>qa_service.py</a> composes answers from the best one.
            Multi-shape figures use the fixed slots from <a href={blob('level-3-deep/deep_eval.py')}>deep_eval.py</a>.
          </p>
        </div>
        {model ? <Live>{`live encoder: ${model.steps} steps, seed ${model.seed}`}</Live> : <span className="pill bad">● untrained encoder: run the Trainer first</span>}
      </div>
      <div className="grid2">
        <div>
          <div className="controls" style={{ gridTemplateColumns: 'repeat(2, minmax(0, 1fr))' }}>
            <Select label="Colour" value={colour} options={L.COLOURS} onChange={setColour} />
            <Select label="Shape" value={shape} options={L.SHAPES} onChange={setShape} />
            <Select label="Shapes in figure" value={String(items)} options={['1', '2', '3', '4']} onChange={(v) => setItems(+v)} />
            <Select label="Paint brush" value={brush} options={BRUSHES} onChange={setBrush} />
            {items === 1 && <>
              <Range label="Jitter x" value={dx} min={-3} max={3} step={1} onChange={setDx} show={dx} />
              <Range label="Jitter y" value={dy} min={-3} max={3} step={1} onChange={setDy} show={dy} />
              <Range label="Size" value={size} min={4} max={5} step={1} onChange={setSize} show={size} />
            </>}
          </div>
          <svg className="chart paint" viewBox="0 0 160 160" role="img" aria-label="Paintable 16 by 16 input image" onPointerLeave={() => {}}>
            {Array.from({ length: L.IMAGE_SIZE * L.IMAGE_SIZE }, (_, i) => (
              <rect key={i} x={(i % L.IMAGE_SIZE) * 10} y={Math.floor(i / L.IMAGE_SIZE) * 10} width="9.6" height="9.6" rx="1.5" fill={rgbCss(img, i)}
                onPointerDown={(e) => { e.preventDefault(); paint(i); }} onPointerEnter={(e) => { if (e.buttons === 1) paint(i); }} />
            ))}
          </svg>
          <p className="hint">Click or drag on the grid to paint with the brush. Changing a control redraws the figure.</p>
        </div>
        <div>
          <h3>Caption similarity (cosine)</h3>
          <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Cosine similarity of the image to each of the 12 captions">
            <line className="axis" x1={sx(0)} x2={sx(0)} y1={0} y2={H - 8} />
            {pred.sims.map((s, j) => {
              const best = L.COMBOS[j] === `${pred.colour} ${pred.shape}`;
              return (
                <g key={j} transform={`translate(0 ${j * bh + 4})`}>
                  <text x={padL - 8} y={bh * 0.65} textAnchor="end" style={best ? { fill: 'var(--text)', fontWeight: 700 } : null}>a {L.COMBOS[j]}</text>
                  <rect x={Math.min(sx(0), sx(s))} width={Math.abs(sx(s) - sx(0))} y={2} height={bh - 6} rx="3" fill={best ? 'var(--primary)' : 'var(--grid)'} />
                  <text x={Math.max(sx(0), sx(s)) + 4} y={bh * 0.65}>{s.toFixed(2)}</text>
                </g>
              );
            })}
          </svg>
          <h3 style={{ marginTop: 12 }}>Service answers</h3>
          <div style={{ overflowX: 'auto' }}>
            <table>
              <thead><tr><th>Question</th><th>Answer</th></tr></thead>
              <tbody>
                {SKETCH_QS.map((qt) => {
                  const a = L.answerQuestion(params, img, L.QUESTIONS[qt], pred).answer;
                  const wrong = qt === 'count' && goldCount && goldCount > 1;
                  return <tr key={qt}><td>{L.QUESTIONS[qt]}</td><td className={wrong ? 'no' : ''}>{a}{wrong ? ` (figure has ${L.NUM_WORDS[goldCount]})` : ''}</td></tr>;
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </section>
  );
}

// ---------- Build ----------
function Build() {
  const [qt, setQt] = useState('all');
  const [size, setSize] = useState(L.IMAGE_SIZE);
  const [patch, setPatch] = useState(build.tokenization.patch_size);
  const rows = build.per_case.filter((r) => qt === 'all' || r.question_type === qt);
  const vt = L.visualTokenCount(size, patch);
  const unified = Object.entries(L.BUILD_QUESTIONS).map(([k, q]) => [k, L.textTokenCount(q), L.unifiedTokenCount(q, size, patch)]);
  return (
    <section id="build" className="card window reveal" data-title="qa_service.py - Build">
      <div className="head">
        <div>
          <p className="eyebrow">Level 2 · Build</p>
          <h2>A schema-validated QA service, scored case by case.</h2>
          <p className="sub">
            {build.eval_cases} synthetic cases, one image per colour and shape combination, four question types. Exact match {f3(build.exact_match)}, token F1 {f3(build.mean_token_f1)}, schema {build.schema_pass} of {build.eval_cases}.
            The perfect score proves the service loop on one image per combination, not VLM skill.
          </p>
        </div>
        <Recorded>recorded, results_build.json</Recorded>
      </div>
      <div className="seg" role="group" aria-label="Filter by question type" style={{ marginBottom: 12 }}>
        {['all', ...Object.keys(build.per_question_type)].map((k) => <button key={k} aria-pressed={qt === k} onClick={() => setQt(k)}>{k}</button>)}
      </div>
      <div className="scroll-box" style={{ overflow: 'auto', maxHeight: 300 }}>
        <table>
          <thead><tr><th>Case</th><th>Gold</th><th>Predicted</th><th>EM</th><th>F1</th><th>Schema</th></tr></thead>
          <tbody>
            {rows.map((r) => <tr key={r.case_id}><td className="mono">{r.case_id}</td><td>{r.labelled_answer}</td><td>{r.predicted_answer}</td><td>{f3(r.exact_match)}</td><td>{f3(r.token_f1)}</td><td><span className={`pill ${r.schema_valid ? 'good' : 'bad'}`}>{r.schema_valid ? 'valid' : 'invalid'}</span></td></tr>)}
          </tbody>
        </table>
      </div>
      <p className="hint">{rows.length} rows shown. Exact match on this filter: {f3(L.mean(rows.map((r) => r.exact_match)))} (computed here from the rows).</p>
      <h3 style={{ marginTop: 20 }}>Tokenization calculator <Live /></h3>
      <p className="hint" style={{ marginTop: 4 }}>Same formulas as <code>visual_token_count</code> and <code>text_token_count</code>: (size ÷ patch)² patches plus one class token, then one token per question word. The repo's defaults give {build.tokenization.visual_tokens_with_class_token} visual tokens.</p>
      <div className="grid2">
        <div className="controls" style={{ gridTemplateColumns: '1fr' }}>
          <Range label="Image size (pixels per side)" value={size} min={8} max={64} step={4} onChange={setSize} show={size} />
          <Range label="Patch size" value={patch} min={1} max={16} step={1} onChange={setPatch} show={patch} />
        </div>
        <div>
          <div className="tiles" style={{ gridTemplateColumns: '1fr 1fr', marginTop: 0 }}>
            <div className="tile"><div className="num">{vt.with_class_token}</div><div className="cap">visual tokens ({vt.patches} patches + class)</div></div>
            <div className="tile"><div className="num">{Math.min(...unified.map((u) => u[2]))}-{Math.max(...unified.map((u) => u[2]))}</div><div className="cap">unified tokens per question</div></div>
          </div>
          <p className="hint">{unified.map(([k, t, u]) => `${k}: ${vt.with_class_token} + ${t} = ${u}`).join(' · ')}</p>
        </div>
      </div>
    </section>
  );
}

// ---------- Deep study ----------
function Heatmap({ rows, route }) {
  const docs = [...new Set(rows.map((r) => r.doc_id))];
  const cw = 34, ch = 30, padL = 78, padT = 30, W = padL + docs.length * cw, H = padT + L.QUESTION_TYPES.length * ch + 4;
  const [hover, setHover] = useState(null);
  return (
    <>
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Exact match heatmap, ${route} route, 24 documents by 6 question types`} onMouseLeave={() => setHover(null)}>
        {docs.map((d, i) => <text key={d} x={padL + i * cw + cw / 2} y={padT - 18} textAnchor="middle">{d.slice(1)}</text>)}
        {docs.map((d, i) => <text key={d + 'n'} x={padL + i * cw + cw / 2} y={padT - 5} textAnchor="middle" style={{ fontSize: 9 }}>{'•'.repeat(rows.find((r) => r.doc_id === d).item_count)}</text>)}
        {L.QUESTION_TYPES.map((qt, j) => <text key={qt} x={padL - 8} y={padT + j * ch + ch / 2 + 4} textAnchor="end">{qt}</text>)}
        {rows.map((r) => {
          const i = docs.indexOf(r.doc_id), j = L.QUESTION_TYPES.indexOf(r.question_type), em = r[`${route}_exact_match`], f1 = r[`${route}_token_f1`];
          return <rect key={r.case_id} x={padL + i * cw + 1.5} y={padT + j * ch + 1.5} width={cw - 3} height={ch - 3} rx="4"
            fill={em ? 'var(--good)' : f1 > 0 ? 'var(--warn)' : 'var(--bad)'} opacity={hover && hover.case_id !== r.case_id ? 0.45 : 0.85} onMouseEnter={() => setHover(r)} />;
        })}
      </svg>
      <div className="tip" aria-live="polite">
        {hover ? <><span className="mono">{hover.case_id}</span> ({hover.item_count} item{hover.item_count > 1 ? 's' : ''}): gold “{hover.labelled_answer}”, vision “{hover.vision_answer}”, text “{hover.text_answer}”</> : 'Hover a cell to see both answers. Dots above each column give the item count.'}
      </div>
      <div className="legend"><span><i style={{ background: 'var(--good)' }} />exact match</span><span><i style={{ background: 'var(--warn)' }} />partial (token F1 &gt; 0)</span><span><i style={{ background: 'var(--bad)' }} />wrong</span></div>
    </>
  );
}

function Study({ model }) {
  const [route, setRoute] = useState('vision');
  const [liveRows, setLiveRows] = useState(null);
  const rec = useMemo(() => L.byType(deep.per_case), []);
  const live = useMemo(() => liveRows && L.byType(liveRows), [liveRows]);
  const groups = useMemo(() => L.QUESTION_TYPES.map((qt) => {
    const r = deep.per_case.filter((x) => x.question_type === qt);
    return [qt, L.mean(r.filter((x) => x.item_count === 1).map((x) => x.vision_exact_match)), L.mean(r.filter((x) => x.item_count > 1).map((x) => x.vision_exact_match))];
  }), []);
  const total = (rows, k) => L.mean(rows.map((r) => r[k]));
  return (
    <section id="study" className="card window reveal" data-title="deep_eval.py - Study">
      <div className="head">
        <div>
          <p className="eyebrow">Level 3 · Deep</p>
          <h2>Vision wins only where the answer is <em>in the pixels</em>.</h2>
          <p className="sub">
            {deep.documents} synthetic documents, each a {L.IMAGE_SIZE}×{L.IMAGE_SIZE} figure plus a text layer of the form <code>{L.textLayer('alpha', [['red', 'circle'], ['green', 'square']])}</code>.
            The text layer names the shapes in order but has no colour words, so colour lives only in the figure. That design favours the parser on most question types, which the repo states openly.
          </p>
        </div>
        <div className="seg" role="group" aria-label="Route shown in heatmap">
          {['vision', 'text'].map((k) => <button key={k} aria-pressed={route === k} onClick={() => setRoute(k)}>{k} route</button>)}
        </div>
      </div>
      <Heatmap rows={deep.per_case} route={route} />
      <p className="hint"><Recorded>recorded, results_deep.json</Recorded> {deep.eval_cases} cases. The table below is computed in your browser from those rows.</p>
      <div className="actions" style={{ marginBottom: 12 }}>
        <button className="btn primary" disabled={!model} onClick={() => setLiveRows(L.runDeep(model.params))}>Rerun study live</button>
        <span className="hint">{model ? `Renders the ${deep.documents} documents, runs the text parser and the QA service on your ${model.steps}-step encoder.` : 'Train the encoder in the Trainer window first; the rerun uses it as the vision route.'}</span>
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table>
          <thead><tr><th>Question type</th><th>Vision EM</th><th>Vision F1</th><th>Text EM</th><th>Text F1</th>{live && <><th>Live vision EM</th><th>Live text EM</th></>}</tr></thead>
          <tbody>
            {L.QUESTION_TYPES.map((qt) => (
              <tr key={qt}><td>{qt}</td><td>{f3(rec[qt].vision_exact_match)}</td><td>{f3(rec[qt].vision_mean_token_f1)}</td><td>{f3(rec[qt].text_exact_match)}</td><td>{f3(rec[qt].text_mean_token_f1)}</td>
                {live && <><td>{f3(live[qt].vision_exact_match)}</td><td className={live[qt].text_exact_match === rec[qt].text_exact_match ? 'ok' : 'no'}>{f3(live[qt].text_exact_match)}</td></>}</tr>
            ))}
            <tr><td><b>overall ({deep.eval_cases})</b></td><td>{f3(total(deep.per_case, 'vision_exact_match'))}</td><td>{f3(total(deep.per_case, 'vision_token_f1'))}</td><td>{f3(total(deep.per_case, 'text_exact_match'))}</td><td>{f3(total(deep.per_case, 'text_token_f1'))}</td>
              {live && <><td>{f3(total(liveRows, 'vision_exact_match'))}</td><td className="ok">{f3(total(liveRows, 'text_exact_match'))}</td></>}</tr>
          </tbody>
        </table>
      </div>
      {live && <p className="hint">Live columns: your browser. The text route is deterministic, so its live column must equal the recorded one exactly (green). The vision column depends on your encoder and RNG.</p>}
      <h3 style={{ marginTop: 20 }}>Vision exact match, single-item vs multi-item documents</h3>
      <div className="bars" style={{ marginTop: 10 }}>
        {groups.map(([qt, s, m]) => (
          <div className="bar-row" key={qt}>
            <div className="lab"><span>{qt}</span><b>{f3(s)} single · {f3(m)} multi</b></div>
            <div className="track split"><div className="fill" style={{ width: `${s * 50}%`, background: 'var(--primary)' }} /><div className="fill" style={{ width: `${m * 50}%`, background: 'var(--warn)', left: '50%', position: 'absolute', top: 0 }} /></div>
          </div>
        ))}
      </div>
      <div className="legend"><span><i style={{ background: 'var(--primary)' }} />single-item (left half)</span><span><i style={{ background: 'var(--warn)' }} />multi-item (right half)</span></div>
    </section>
  );
}

// ---------- Counting failure class ----------
function Counting() {
  const rows = deep.per_case.filter((r) => r.question_type === 'count');
  const byN = [1, 2, 3, 4].map((n) => { const r = rows.filter((x) => x.item_count === n); return { n, cases: r.length, v: L.mean(r.map((x) => x.vision_exact_match)), t: L.mean(r.map((x) => x.text_exact_match)), ans: [...new Set(r.map((x) => x.vision_answer))].join(', ') }; });
  const W = 460, H = 220, pad = { l: 36, r: 10, t: 14, b: 40 }, gw = (W - pad.l - pad.r) / 4, bw = gw / 4;
  const y = (v) => pad.t + (1 - v) * (H - pad.t - pad.b);
  const overall = deep.per_question_type.count.vision_exact_match;
  return (
    <section id="counting" className="card window reveal" data-title="counting.md - Failure class">
      <p className="eyebrow">Failure class</p>
      <h2>The vision route says “one” every time, so it is right only when there is one.</h2>
      <div className="grid2" style={{ marginTop: 16 }}>
        <div>
          <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Count question exact match by gold item count, vision and text routes">
            {[0, 0.5, 1].map((t) => <g key={t}><line className="axis" x1={pad.l} x2={W - pad.r} y1={y(t)} y2={y(t)} /><text x={pad.l - 6} y={y(t) + 4} textAnchor="end">{t}</text></g>)}
            {byN.map((g, i) => (
              <g key={g.n} transform={`translate(${pad.l + i * gw} 0)`}>
                <rect x={bw * 0.4} y={y(g.v)} width={bw} height={Math.max(1.5, y(0) - y(g.v))} rx="3" fill="var(--primary)" />
                <rect x={bw * 1.5} y={y(g.t)} width={bw} height={y(0) - y(g.t)} rx="3" fill="var(--good)" />
                <rect x={bw * 2.6} y={y(countBase.exact_match_by_gold_items[g.n])} width={bw} height={Math.max(1.5, y(0) - y(countBase.exact_match_by_gold_items[g.n]))} rx="3" fill="var(--warn)" />
                <text x={gw / 2} y={H - 22} textAnchor="middle" style={{ fill: 'var(--text)' }}>{g.n} item{g.n > 1 ? 's' : ''}</text>
                <text x={gw / 2} y={H - 8} textAnchor="middle">vision said “{g.ans}”</text>
              </g>
            ))}
          </svg>
          <div className="legend"><span><i style={{ background: 'var(--primary)' }} />vision exact match</span><span><i style={{ background: 'var(--good)' }} />text exact match</span><span><i style={{ background: 'var(--warn)' }} />region-count baseline</span></div>
          <p className="hint">All bars recorded: <code>results_deep.json</code> and <code>analysis/results_count_baseline.json</code>.</p>
        </div>
        <div>
          <p className="sub" style={{ marginTop: 0 }}>
            Count exact match is {f3(overall)} overall: exactly the {byN[0].cases} single-item documents out of {rows.length}. On multi-item documents it is {f3(deep.vision_by_item_group.count_multi.vision_exact_match)}.
            This is not an empirical surprise but a property of the code: the composer has no enumeration step. The lines below are read from <a href={blob('level-2-build/qa_service.py')}>qa_service.py</a> at build time.
          </p>
          <pre className="code">{countBranch}</pre>
          <p className="hint">The text parser gets {f3(deep.per_question_type.count.text_exact_match)} because the legend lists one shape name per item.
            A new enumeration baseline, <a href={blob('analysis/count_baseline.py')}>count_baseline.py</a>, simply counts connected non-background regions: {f3(countBase.exact_match)} exact match with no learning. So the pixels carry most counts. It fails on every three-item figure because those slots draw neighbouring shapes touching, a flaw in the document set itself. A learned enumeration step in the service remains v1.1 backlog item 3.</p>
        </div>
      </div>
    </section>
  );
}

// ---------- Uncertainty ----------
function Uncertainty() {
  const [B, setB] = useState(2000);
  const rows = useMemo(() => {
    const out = L.QUESTION_TYPES.map((qt) => {
      const r = deep.per_case.filter((x) => x.question_type === qt);
      const v = r.map((x) => x.vision_exact_match), t = r.map((x) => x.text_exact_match);
      return { qt, n: r.length, v: L.mean(v), t: L.mean(t), vci: L.bootstrapCI(v.map((x) => [x]), B), tci: L.bootstrapCI(t.map((x) => [x]), B), m: L.mcnemarExact(v, t) };
    });
    const docs = [...new Set(deep.per_case.map((r) => r.doc_id))], by = (k) => docs.map((d) => deep.per_case.filter((r) => r.doc_id === d).map((r) => r[k]));
    out.push({ qt: 'overall', n: deep.per_case.length, v: deep.overall.vision_exact_match, t: deep.overall.text_exact_match, vci: L.bootstrapCI(by('vision_exact_match'), B), tci: L.bootstrapCI(by('text_exact_match'), B),
      m: L.mcnemarExact(deep.per_case.map((x) => x.vision_exact_match), deep.per_case.map((x) => x.text_exact_match)) });
    return out;
  }, [B]);
  const W = 460, rowH = 30, H = rows.length * rowH + 34, l = 84, r = 12, x = (v) => l + v * (W - l - r);
  return (
    <section id="stats" className="card window reveal" data-title="stats.js - Uncertainty">
      <div className="head">
        <div>
          <p className="eyebrow">New analysis</p>
          <h2>With {rows[0].n} cases per type, are the route differences real?</h2>
          <p className="sub">
            Computed here, not in the repo's v1.0 write-up: percentile bootstrap 95% intervals on exact match and the exact McNemar test on the paired outcomes (each case answered by both routes).
            Inputs are the recorded per-case rows. <a href={blob('analysis/paired_stats.py')}>analysis/paired_stats.py</a> now records the same test ({stats.bootstrap_resamples.toLocaleString()} resamples) so the review can cite it.
          </p>
        </div>
        <Live>computed in your browser from recorded rows</Live>
      </div>
      <div className="controls"><Range label="Bootstrap resamples" value={B} min={200} max={10000} step={200} onChange={setB} show={B.toLocaleString()} /></div>
      <div className="grid2">
        <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Bootstrap 95% intervals on exact match by question type, vision and text">
          {[0, 0.5, 1].map((t) => <g key={t}><line className="axis" x1={x(t)} x2={x(t)} y1={0} y2={H - 20} /><text x={x(t)} y={H - 6} textAnchor="middle">{t}</text></g>)}
          {rows.map((row, i) => (
            <g key={row.qt} transform={`translate(0 ${i * rowH + 20})`}>
              <text x={l - 8} y={4} textAnchor="end" style={{ fill: 'var(--text)' }}>{row.qt}</text>
              <line x1={x(row.vci[0])} x2={x(row.vci[1])} y1={-4} y2={-4} stroke="var(--primary)" strokeWidth="4" strokeLinecap="round" opacity="0.5" />
              <circle cx={x(row.v)} cy={-4} r="4" fill="var(--primary)" />
              <line x1={x(row.tci[0])} x2={x(row.tci[1])} y1={6} y2={6} stroke="var(--good)" strokeWidth="4" strokeLinecap="round" opacity="0.5" />
              <circle cx={x(row.t)} cy={6} r="4" fill="var(--good)" />
            </g>
          ))}
        </svg>
        <div style={{ overflowX: 'auto' }}>
          <table>
            <thead><tr><th>Type</th><th>Vision only right</th><th>Text only right</th><th>McNemar p</th></tr></thead>
            <tbody>{rows.map((row) => <tr key={row.qt}><td>{row.qt}</td><td>{row.m.vision_only_correct}</td><td>{row.m.text_only_correct}</td><td className={row.m.p_value < 0.05 ? 'ok' : ''}>{fmtP(row.m.p_value)}</td></tr>)}</tbody>
          </table>
        </div>
      </div>
      <div className="legend"><span><i style={{ background: 'var(--primary)' }} />vision EM, 95% CI</span><span><i style={{ background: 'var(--good)' }} />text EM, 95% CI</span></div>
      <p className="hint">Per-type intervals resample the {rows[0].n} cases of that type (one per document). The overall interval resamples whole documents, since the six cases of a document are not independent; the overall McNemar p treats cases as independent, so read it as optimistic. A zero-width text interval means the parser is right on every resample. Every route difference is significant here, but only for this synthetic set and this stand-in.</p>
    </section>
  );
}

// ---------- Seeds and confusion ----------
function Seeds() {
  const ls = seeds.learn_summary, vs = seeds.deep_vision_em_summary, conf = seeds.learn_confusion_seed_42;
  return (
    <section id="seeds" className="card window reveal" data-title="run_seeds.py - Seed sensitivity">
      <div className="head">
        <div>
          <p className="eyebrow">New analysis · recorded</p>
          <h2>v1.0 used one seed. Here are {seeds.seeds.length}.</h2>
          <p className="sub">
            <a href={blob('analysis/run_seeds.py')}>analysis/run_seeds.py</a> reruns the Learn and Deep procedures with encoder seeds {seeds.seeds.join(', ')} (seed {learn.master_seed} reproduces the recorded runs exactly). Only the vision route varies; the text parser is deterministic.
          </p>
        </div>
        <Recorded>recorded, results_seeds.json</Recorded>
      </div>
      <div className="tiles">
        <div className="tile"><div className="num">{f3(ls.image_to_text_recall_at_1.mean)}</div><div className="cap">Learn image→text R@1, mean (sd {f3(ls.image_to_text_recall_at_1.sd)}, range {f3(ls.image_to_text_recall_at_1.min)} to {f3(ls.image_to_text_recall_at_1.max)})</div></div>
        <div className="tile"><div className="num">{f3(ls.text_to_image_recall_at_1.mean)}</div><div className="cap">Learn text→image R@1, mean (sd {f3(ls.text_to_image_recall_at_1.sd)})</div></div>
        <div className="tile"><div className="num">{f3(vs.overall.mean)}</div><div className="cap">Deep vision EM, mean (sd {f3(vs.overall.sd)}, range {f3(vs.overall.min)} to {f3(vs.overall.max)})</div></div>
        <div className="tile"><div className="num">{f3(vs.count.sd)}</div><div className="cap">sd of vision count EM across seeds: constant by construction</div></div>
      </div>
      <div className="grid2" style={{ marginTop: 16 }}>
        <div style={{ overflowX: 'auto' }}>
          <table>
            <thead><tr><th>Seed</th><th>i→t R@1</th><th>t→i R@1</th><th>Deep vision EM</th></tr></thead>
            <tbody>{seeds.learn_per_seed.map((r, i) => <tr key={r.seed}><td>{r.seed}{r.seed === learn.master_seed ? ' (recorded)' : ''}</td><td>{f3(r.image_to_text_recall_at_1)}</td><td>{f3(r.text_to_image_recall_at_1)}</td><td>{f3(seeds.deep_vision_em_per_seed[i].overall)}</td></tr>)}</tbody>
          </table>
        </div>
        <div>
          <h3>Why image→text R@1 is {f3(learn.after_training.image_to_text_recall_at_1)} at seed {conf.seed}</h3>
          <ul className="list">
            {conf.errors.map((e) => <li key={e.gold}>{e.count} held-out image of a <b>{e.gold}</b> retrieved “a {e.predicted}”</li>)}
          </ul>
          <p className="hint">{conf.note}</p>
        </div>
      </div>
    </section>
  );
}

// ---------- Anatomy and roadmap ----------
function Anatomy() {
  return (
    <section id="anatomy" className="card window reveal" data-title="ANATOMY.md - LLaVA">
      <p className="eyebrow">Document inspection, no run</p>
      <h2>Where a real VLM would break: the joints between its parts.</h2>
      <p className="sub">Read from <a href={blob('level-1-learn/ANATOMY.md')}>ANATOMY.md</a>, a description of the published LLaVA design. No VLM was downloaded or run, and nothing on this card is a measurement.</p>
      <div className="grid2" style={{ marginTop: 16 }}>
        {anatomy.map((p) => (
          <div className="tile" key={p.title}>
            <h3>Part {p.title}</h3>
            <p className="hint" style={{ margin: '6px 0 0' }}><b>Job.</b> {p.job}</p>
            <p className="hint" style={{ margin: '6px 0 0' }}><b>Failure tendency.</b> {p.fail}</p>
          </div>
        ))}
      </div>
      <div className="card window inner" data-title="ROADMAP.md - v1.1" id="roadmap">
        <p className="eyebrow">Not built, not measured</p>
        <ol className="steps">
          {backlog.slice(0, 4).map((b) => <li key={b.n}><span className="n">{b.n}</span><div className="hint">{b.text}</div></li>)}
        </ol>
        <p className="hint" style={{ marginTop: 12 }}>Grades on record, self-assessed with no external review: Learn {grades.Learn}, Build {grades.Build}, Deep {grades.Deep} (of 10,000).</p>
      </div>
    </section>
  );
}

function Limits() {
  const textCovered = Object.values(deep.per_question_type).filter((v) => v.text_exact_match === 1).length;
  const items = [
    ['No real VLM', `The vision route is the Learn encoder plus a keyword composer (${L.SOURCE}). The original question, VLM against text pipeline, is still open.`],
    ['Synthetic, designed data', `The text layer answers ${textCovered} of ${L.QUESTION_TYPES.length} question types by design, so the overall text win is partly built in.`],
    ['Counting is a code property', `Count is constant one in the composer, so the failure is true by construction. A region-count baseline added here scores ${f3(countBase.exact_match)}; the service itself still has no enumeration step.`],
    ['Weak generalisation probe', 'Held-out pairs come from the same generator, varying only position and size.'],
    ['Small samples', `${deep.per_question_type.colour.cases} cases per question type. Intervals above are wide; the overall McNemar test ignores document clustering.`],
    ['Untested failure classes', 'Small text was not measured, and spatial relations appear in one question type only.'],
  ];
  return (
    <section id="limits" className="card window reveal" data-title="LIMITATIONS.md - Threats to validity">
      <p className="eyebrow">Limitations and threats to validity</p>
      <h2>What this project does not show.</h2>
      <div className="grid2" style={{ marginTop: 16 }}>
        {items.map(([t, d]) => <div className="tile" key={t}><div className="head" style={{ marginBottom: 6 }}><h3>{t}</h3><span className="pill warn">Caveat</span></div><p className="hint" style={{ margin: 0, fontSize: 13 }}>{d}</p></div>)}
      </div>
      <p className="hint" style={{ marginTop: 16 }}>Full grading against a research rubric, with what each remaining gap needs: <a href={blob('docs/research-review.md')}>docs/research-review.md</a>.</p>
    </section>
  );
}
