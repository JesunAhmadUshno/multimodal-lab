// Known-answer self-check for every computation the page runs: node src/lab.check.mjs (runs in npm run build).
// Deterministic ports must reproduce the repo's recorded per-case results exactly; the trainer must pass
// the repo's own oracles (tests/test_learn_multimodal.py) and finite-difference gradient check.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import * as L from './lab.js';

const root = new URL('../../', import.meta.url);
const json = (p) => JSON.parse(fs.readFileSync(new URL(p, root), 'utf8'));
const learn = json('level-1-learn/results_learn.json'), build = json('level-2-build/results_build.json');
const countBase = json('analysis/results_count_baseline.json');
const deep = json('level-3-deep/results_deep.json'), stats = json('analysis/results_stats.json'), seeds = json('analysis/results_seeds.json');
const near = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg}: ${a} vs ${b}`);
const r6 = (x) => Math.round(x * 1e6) / 1e6;

// 0. Port constants equal the recorded configuration.
assert.equal(L.IMAGE_SIZE, learn.image_shape[0]); assert.equal(L.EMBED_DIM, learn.embed_dim);
assert.equal(L.TEMPERATURE, learn.temperature); assert.equal(L.VOCAB.length, learn.vocab_size); assert.equal(L.COMBOS.length, learn.combinations);
assert.equal(L.PATCH_SIZE, build.tokenization.patch_size); assert.equal(L.SOURCE, deep.vlm_substitution.split(':')[0]);

// 1. Deep text route and gold labels: the port reproduces all 144 recorded rows exactly.
const live = L.runDeep(null);
assert.equal(live.length, 144);
live.forEach((row, i) => {
  const rec = deep.per_case[i];
  for (const k of ['case_id', 'doc_id', 'item_count', 'question_type', 'labelled_answer', 'text_answer']) assert.equal(row[k], rec[k], `${rec.case_id} ${k}`);
  assert.equal(r6(row.text_exact_match), rec.text_exact_match); assert.equal(r6(row.text_token_f1), rec.text_token_f1);
});
assert.equal(r6(L.mean(live.map((r) => r.text_exact_match))), deep.overall.text_exact_match); // 0.666667
assert.equal(r6(L.mean(live.map((r) => r.text_token_f1))), deep.overall.text_mean_token_f1); // 0.75
assert.equal(deep.overall.text_exact_match, 0.666667); assert.equal(deep.overall.text_mean_token_f1, 0.75);

// 2. In-browser aggregates of the recorded per_case rows equal the recorded summaries.
const agg = L.byType(deep.per_case);
for (const [qt, rec] of Object.entries(deep.per_question_type)) for (const k of Object.keys(rec)) assert.equal(r6(agg[qt][k]), rec[k], `${qt} ${k}`);
for (const [key, rec] of Object.entries(deep.vision_by_item_group)) {
  const [qt, grp] = [key.replace(/_(single|multi)$/, ''), key.match(/(single|multi)$/)[1]];
  const rows = deep.per_case.filter((r) => r.question_type === qt && (grp === 'single') === (r.item_count === 1));
  assert.equal(rows.length, rec.cases); assert.equal(r6(L.mean(rows.map((r) => r.vision_exact_match))), rec.vision_exact_match, key);
}
const bAgg = {}; build.per_case.forEach((r) => { (bAgg[r.question_type] ??= []).push(r.exact_match); });
for (const [qt, rec] of Object.entries(build.per_question_type)) assert.equal(r6(L.mean(bAgg[qt])), rec.exact_match);
assert.equal(build.per_case.filter((r) => r.schema_valid).length, build.schema_pass);

// 3. Build: question routing, scoring and token counts (qa_service.py).
for (const [qt, q] of Object.entries(L.BUILD_QUESTIONS)) { assert.equal(L.questionType(q), qt); assert.equal(L.textTokenCount(q), build.tokenization.question_text_tokens[qt]); }
assert.equal(L.questionType(L.QUESTIONS.title), 'unsupported'); assert.equal(L.questionType(L.QUESTIONS.left_shape), 'shape');
assert.equal(L.visualTokenCount().patches, build.tokenization.image_patches); // 16
assert.equal(L.visualTokenCount().with_class_token, build.tokenization.visual_tokens_with_class_token); // 17
assert.equal(L.unifiedTokenCount(L.BUILD_QUESTIONS.colour), build.tokenization.unified_tokens_example_colour_question); // 22
assert.deepEqual(Object.values(L.BUILD_QUESTIONS).map((q) => L.unifiedTokenCount(q)).sort(), [22, 23, 24, 25]);
assert.equal(L.tokenF1('unknown circle', 'red circle'), 0.5); assert.equal(L.exactMatch(' Red  ', 'red'), 1);

// 4. Statistics: exact McNemar equals analysis/paired_stats.py; bootstrap CIs agree within resampling noise.
for (const qt of L.QUESTION_TYPES) {
  const r = deep.per_case.filter((x) => x.question_type === qt), rec = stats.per_question_type[qt];
  const m = L.mcnemarExact(r.map((x) => x.vision_exact_match), r.map((x) => x.text_exact_match));
  assert.equal(m.vision_only_correct, rec.mcnemar.vision_only_correct); assert.equal(m.text_only_correct, rec.mcnemar.text_only_correct);
  near(m.p_value, rec.mcnemar.p_value, 1e-12, `mcnemar ${qt}`);
  const ci = L.bootstrapCI(r.map((x) => [x.vision_exact_match]), 10000);
  near(ci[0], rec.vision_em_ci95[0], 0.05, `ci lo ${qt}`); near(ci[1], rec.vision_em_ci95[1], 0.05, `ci hi ${qt}`);
}
const mAll = L.mcnemarExact(deep.per_case.map((x) => x.vision_exact_match), deep.per_case.map((x) => x.text_exact_match));
near(mAll.p_value, stats.overall.mcnemar.p_value, 1e-15, 'mcnemar overall');
assert.equal(L.mcnemarExact([1, 1, 1, 0], [0, 0, 0, 0]).p_value, 0.25);

// 5. Learn trainer: the repo's loss oracles and a finite-difference gradient check.
const eye = new Float64Array([1, 0, 0, 1]);
near(L.contrastiveLoss(eye, eye, 2, 1.0, 2), 0.3132616875182228, 1e-9, 'oracle identity t=1');
near(L.contrastiveLoss(new Float64Array([1, 0, 1, 0]), new Float64Array([1, 0, 1, 0]), 2, 1.0, 2), Math.log(2), 1e-9, 'oracle uniform');
near(L.contrastiveLoss(eye, eye, 2, 0.5, 2), 0.1269280110429725, 1e-9, 'oracle t=0.5');
{
  const P = L.initParams(), d = L.makePairs(1, 3000), X = d.images.subarray(0, 4 * L.PIX), T = d.tokens.slice(0, 4);
  const { grads } = L.lossAndGrads(P, X, T), rand = L.mulberry32(7), eps = 1e-6;
  for (const k of ['W1', 'W2', 'E', 'Wt', 'b1', 'bt']) for (let s = 0; s < 6; s++) {
    const i = Math.floor(rand() * P[k].length), old = P[k][i];
    P[k][i] = old + eps; const lp = L.lossAndGrads(P, X, T).loss; P[k][i] = old - eps; const lm = L.lossAndGrads(P, X, T).loss; P[k][i] = old;
    const num = (lp - lm) / (2 * eps);
    assert.ok(Math.abs(num - grads[k][i]) / Math.max(1, Math.abs(num), Math.abs(grads[k][i])) < 1e-4, `grad ${k}[${i}] ${num} vs ${grads[k][i]}`);
  }
  // embedding loss equals the loss inside lossAndGrads
  const I = L.encodeImage(P, X, 4), Tz = L.encodeText(P, T);
  near(L.contrastiveLoss(I, Tz, 4), L.lossAndGrads(P, X, T).loss, 1e-12, 'loss consistency');
}

// 6. Live retrain at the page default (1,000 steps) learns the task the way the recorded run did.
const t0 = Date.now(), train = L.makePairs(100, 100000), held = L.makePairs(10, 900000);
const before = L.evaluate(L.initParams(42), held), tr = L.makeTrainer(train, 42);
tr.step(1000);
const after = L.evaluate(tr.params, held), lossFirst = tr.losses[0], lossLast = tr.losses.at(-1);
assert.ok(lossLast < 0.05 * lossFirst, `loss ${lossFirst} -> ${lossLast}`);
assert.ok(after.image_to_text_recall_at_1 >= 0.9 && after.text_to_image_recall_at_1 >= 0.9, JSON.stringify(after));
// R@5 equals R@1 image-to-text by construction (identical captions), as analysis/results_seeds.json notes
assert.equal(after.image_to_text_recall_at_5, after.image_to_text_recall_at_1);
assert.ok(seeds.learn_confusion_seed_42.note.includes('by construction'));
// Vision route on the live encoder keeps the failure class: count is 'one' on every multi-item document.
const liveDeep = L.runDeep(tr.params);
assert.ok(liveDeep.filter((r) => r.question_type === 'count' && r.item_count > 1).every((r) => r.vision_answer === 'one'));
assert.ok(liveDeep.filter((r) => r.question_type === 'title').every((r) => r.vision_answer === 'unsupported question'));

// 7. Recorded imports equal the benchmark table kept in REPORT.md (Appendix B, former README).
const readme = fs.readFileSync(new URL('REPORT.md', root), 'utf8');
for (const [label, value] of [
  ['Learn train loss, first step to last step', `${learn.train_loss_first} to ${learn.train_loss_last}`],
  ['Learn image-to-text Recall at 1, before to after', `${learn.before_training.image_to_text_recall_at_1.toFixed(6)} to ${learn.after_training.image_to_text_recall_at_1.toFixed(6)}`],
  ['Learn text-to-image Recall at 1, before to after', `${learn.before_training.text_to_image_recall_at_1.toFixed(6)} to ${learn.after_training.text_to_image_recall_at_1.toFixed(6)}`],
  ['Build exact match, 48 synthetic QA cases', build.exact_match.toFixed(3)],
  ['Build schema validation', `${build.schema_pass} of ${build.eval_cases} passed`],
  ['Deep vision exact match, 144 synthetic document cases', deep.overall.vision_exact_match.toFixed(6)],
  ['Deep text-only exact match, same 144 cases', deep.overall.text_exact_match.toFixed(6)],
  ['Deep vision count exact match (text 1.000)', deep.per_question_type.count.vision_exact_match.toFixed(3)],
]) assert.ok(readme.includes(`| ${label} | ${value} |`), `README row ${label} = ${value}`);

// REPORT.md post-v1.0 section cites analysis/results_*.json
const f3 = (x) => x.toFixed(3);
for (const frag of [
  `Recall at 1 is ${f3(seeds.learn_summary.image_to_text_recall_at_1.mean)} mean (sd ${f3(seeds.learn_summary.image_to_text_recall_at_1.sd)}`,
  `Deep vision exact match ${f3(seeds.deep_vision_em_summary.overall.mean)} mean (sd ${f3(seeds.deep_vision_em_summary.overall.sd)}`,
  `largest p ${Math.max(...Object.values(stats.per_question_type).map((v) => v.mcnemar.p_value)).toFixed(4)}`,
  `scores ${f3(countBase.exact_match)} exact match on the ${countBase.cases} count questions`,
  `CI of ${f3(stats.overall.vision_em_ci95[0])} to ${f3(stats.overall.vision_em_ci95[1])}`,
]) assert.ok(readme.includes(frag), `README cites ${frag}`);

console.log('lab ok', { text_em: deep.overall.text_exact_match, live_steps: tr.steps, live_after: after, ms: Date.now() - t0 });
