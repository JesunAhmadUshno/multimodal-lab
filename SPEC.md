# SPEC: multimodal-lab (P22)

Phase skill: spec-driven-development (house engineering pipeline).
Status: Wave 0 scaffold spec, from Roadmap v2 text and the v3 decision pack. Scope and acceptance criteria below are the v2 section text, kept as the source of truth until the project's wave opens its own spec pass.

Fleet node: BlackBox (GPU). Wave: Wave 8.
Hiring evidence: 3 of 5 (1 posting read in full, trend years 2027 to 2028) [confirmed, fleet and scoring pass 2026-11-04, or the decision pack for P23 to P29].

## Project section text (Roadmap v2, or the v3 candidate text for P23 to P29)

multimodal-lab: NEW

Subject: vision-language systems. Evidence: the xAI Multimodal Understanding posting read in full; register items 310, 311, 333 name the builds and sit homeless in v1 [confirmed].
- Learn: train a small CLIP-style dual encoder on public data, once, end to end (register item 310); write the VLM anatomy page (encoder, projector, language model joints) with one open VLM inspected part by part (register item 311).
- Build: wire one open vision-language model into a question-answering service over images, with the P14 schema rules on its outputs and the P06 harness scoring its answers; a multimodal tokenization note with counts (image patches, audio frames, unified token streams; register item 333).
- Deep: an eval study: the VLM against a text-only pipeline on the same document questions from P16 (where vision pays, where it does not, one table); one failure class written up (counting, small text, spatial relations).
Fan-out: Book chapter "Teaching Machines To See". Paper angle: a small VLM eval note (vision against text-only on document questions). Course module: P22-multimodal-lab.

---

# 6. The mapping table: every project against the evidence

Job evidence counts come from the career documentation, Section B (of 56 in-lane postings) and Section D (build ranks) [confirmed]. Register categories use the register's own numbers and names [confirmed]. CEO atoms use the cross-match codes; a scale atom is marked (scale) [confirmed, cross-match Parts 2 to 5]. Trend years name the years the project answers most (Section 7).

| Project | Job-post skill evidence | Gap register categories it closes | CEO atoms it builds | Trend years |
| --- | --- | --- | --- | --- |
| P01 llm-from-scratch | Framework named in 27; LLM work in 33; Section D rank 2 | C3 PyTorch (80, lead), C6 (50), C5 (60), C4 JAX track (20) | Next-15 step 1; first rung toward S1 (scale) | 2026, 2027 |
| P02 tokenizer-lab | Tokenization and data duties at xAI and Mistral; Section D rank 11 | C7 pre-training and data (45, lead) | Next-15 step 14 | 2026 |
| P03 vllm-serving-lab | vLLM named at xAI; vLLM and SGLang internals at Microsoft; Section D rank 3 | C9 inference and serving (55, lead), C13 drill items | Next-15 step 4; first rung toward S10 (scale) | 2026, 2027 |
| P04 cuda-kernels-lab | CUDA or GPU programming in 12 | C11 CUDA and GPU (35, lead), C2 C++ items (65) | Next-15 step 12 | 2026, 2027 |
| P05 rag-foundation | RAG named in 4, required at Apple and Mistral FDE | C7, C12 data pipelines (40), C17 applied delivery (35) | Supports S3 (production code, PARTIAL today) | 2026 |
| P06 rag-eval-harness | Evaluation core work in 22; Section D rank 1 | C14 evaluation (50, lead), C16 paper track (40), C18 portfolio proof (50), C12 | Next-15 steps 2 and 3; the candidate field-shaping artifact (the E3 substitute path); R10 seed | 2026, 2027, 2028 |
| P07 agent-from-scratch | Agents named in 12 | C8 post-training family support (45), C14 agent track, C15 support | Next-15 step 13 (pair with P08) | 2026, 2027 |
| P08 orchestrator-lab | Agents in 12; multi-agent system in Amazon duties | C1 craft items (70), C17, C18 | Next-15 step 13 (pair with P07) | 2027, 2028 |
| P09 mcp-toolkit | MCP in Apple Applied AI preferred list | C1 tooling items, C18 package items (PyPI, items 3 and 4) | S4 support (open artifact the field can adopt) | 2026, 2027 |
| P10 finetune-lab | Post-training duties at Cohere, Mistral, OpenAI; Section D rank 5 | C8 (45, lead), C3 | Next-15 step 7 | 2026, 2027 |
| P11 quantization-lab | Cohere Model Efficiency posting; quantization preferred at xAI | C9, C4 item 232 | Next-15 step 15 | 2026, 2027 |
| P12 llm-observability | Observability a required duty at Mistral; Section D rank 9 | C13 (50), C1 logging and debugging items | S3 support | 2026, 2027, 2028 |
| P13 guardrails-lab | Safety postings: 3 Anthropic, 3 DeepMind read in full | C15 safety (35, lead) | Next-15 step 8 | 2026, 2027, 2028 |
| P14 structured-outputs-lab | Applied roles at OpenAI, Mistral, Apple (production LLM behavior is the job) | C1 item family, C9 support | No next-15 step | 2026 |
| P15 local-first-ai | Few postings name it; value is positioning (Section D rank 16) | C17, C20 presence support (50) | P5 support (products in two or more domains, PARTIAL today) | 2027, 2028 |
| P16 document-intelligence | Document AI preferred at Mistral; FDE roles at Mistral and OpenAI | C12 (40), C1, C17 | P7 support (a launch-shaped applied product) | 2026, 2027 |
| P17 signal-ml-lab | Data-leaning roles; proves the published record runs | C4 (20, lead for TF depth), C5, C1 NumPy items | R5 already HAS; this project makes it runnable and feeds R10 | 2028 (no trend clock; a record proof) |
| P18 voice-agent-lab | xAI Multimodal and Mistral audio work | C6 media items, C1 streaming items | No next-15 step | 2026, 2027 |
| P19 distributed-training-lab | Distributed or large-scale systems in 31; Section C gap 2 | C10 (55, lead), C13, C3 | Next-15 step 6; first rung toward S1 (scale) and S2 (scale) | 2026, 2027, 2028 |
| P20 ml-platform-lab | Kubernetes in 8; Mistral ML Platform depth (operators, CRDs, Kueue) | C13 (50, lead), C9 item 481 | No next-15 step; S2 support skills | 2026, 2027 |
| P21 systems-languages-lab | Go at Apple and Microsoft; Rust at xAI and Microsoft | C2 (65, lead) | S3 support (production code in systems languages) | 2027, 2028 |
| P22 multimodal-lab | xAI Multimodal Understanding, read in full | C6 items 310 and 311, C7 item 333, C12 media items | No next-15 step | 2027, 2028 |

Read of the table: the next-15 CEO atoms map onto nine projects (P01, P02, P03, P04, P06, P07 with P08, P10, P11, P13, P19). The other projects close posting demand and register weight that the atom list does not carry. Both kinds of work stay in the plan, and the table keeps them apart so neither is mistaken for the other.

---

# 7. Trends 2026, 2027, 2028: what the field asks each year, and which repos answer

Ground rule for this section: the 57 postings are the demand signal first [confirmed]. Forward claims come second, each with its source and its tag. Vendor and bank forecasts are [unverified] by rule: no primary report was opened in this run.

## 2026: agents land, thinking costs money, evaluation becomes law-shaped

- The postings already price 2026: frameworks in 27 of 56, distributed systems in 31, evaluation in 22, agents in 12 [confirmed, Section B].
- Gartner's prediction, as reported: 40 percent of enterprise applications will carry task-specific AI agents by the end of 2026, up from under 5 percent in 2025 [unverified]. Source: https://www.aiplusinfo.com/blog/agentic-ai-adoption-statistics-2026/
- Inference is the larger half of compute: about two thirds of AI compute in 2026, up from one third in 2023, and reasoning models make budget control (how long a model thinks, at what cost) a core serving skill [unverified]. Sources: https://zylos.ai/research/2026-01-24-ai-reasoning-models/ and https://eliteai.tools/blog/test-time-compute-scaling-news-2026
- EU general-purpose AI duties carry fining powers from 2 August 2026, and regulators in the EU, US, and UK converge on mandatory evaluation and adversarial testing as the governance tool [confirmed dates; the convergence read is one analysis]. Sources: https://regulome.io/regulations/eu-ai-act and https://autonainews.com/eu-ai-acts-gpai-model-evaluation-rules-draw-industry-pushback/
- Repos that answer 2026: P06 (evaluation), P01 (PyTorch), P07 (agents), P03 (reasoning serving), P13 (safety and artefacts), P10 (post-training), P05 (retrieval).

## 2027: the agent shakeout, governance deadlines, the infra peak

- Gartner's warning, as reported: over 40 percent of agentic AI projects will be canceled by the end of 2027, on cost, unclear value, and weak risk controls [unverified]. Source: https://www.aiplusinfo.com/blog/agentic-ai-adoption-statistics-2026/ The skill that survives a shakeout is evaluation and cost proof, which is P06 and P12 territory.
- Multi-agent work goes mainstream in the same forecast family: one third of agentic implementations combine agents with different skills by 2027 [unverified]. Source: https://www.leadgen-economy.com/blog/agentic-enterprise-ai-agents-operations/ P08 is the repo that answers it.
- EU high-risk (Annex III) conformity lands 2 December 2027 [confirmed]. Source: https://regulome.io/regulations/eu-ai-act P13's governance artefacts stop being a differentiator and start being a job requirement.
- Infrastructure spend peaks in this window on bank forecasts: about 1.2 trillion dollars across five big firms in 2027, with growth slowing after [unverified]. Source: https://www.n884.com/html/infrastructure/478.html Platform skill (P20, P19) is hired while the buildout runs, not after it.
- Physical AI commercial work accelerates from 2027 as actuation and edge costs fall (GlobalData read) [unverified]. Source: https://it-online.co.za/2026/09/21/physical-ai-will-redefine-the-future-of-robotics/ P22's vision-language base is the entry ticket he can hold by then.

## 2028: the software turn, embedded rules, embodied systems

- Morgan Stanley's read, as reported: hyperscaler capex growth falls to about 14 percent in 2028 and the cycle shifts from hardware toward software and applications [unverified]. Source: https://www.thehindubusinessline.com/info-tech/ai-spending-growth-seen-slowing-sharply-by-2028-marking-shift-towards-software-morgan-stanley/article71518970.ece The 2028 premium sits on shipped applications with users, which is C17 weight: P16, P15, and the product side of P05.
- EU high-risk rules for AI embedded in regulated products land 2 August 2028 [confirmed]. Source: https://regulome.io/regulations/eu-ai-act P13 and P17 (an edge-deployed model with artefacts) answer it.
- Agentic front ends: one third of user experiences shift from native apps to agentic front ends by 2028, and about 15 percent of day-to-day work decisions are made autonomously [unverified]. Source: https://www.leadgen-economy.com/blog/agentic-enterprise-ai-agents-operations/ P08 and P12 (orchestration with traces and drift watch) are the repos that speak this language.
- Physical AI keeps compounding: the US physical AI market grows at about 31.7 percent a year across 2026 to 2035 on one research firm's numbers [unverified]. Source: https://www.smartindustry.com/tools-of-transformation/artificial-intelligence/article/55394960/physical-ais-growth-expected-to-explode P22 plus P18 (perception in, voice out) form his embodied base if he ever takes that lane.
- The honest limit of this section: no 2028 posting exists today. The 2028 column is forecast, tagged as such, and the plan treats it as direction, not as demand. The register re-grades against real postings each cycle [confirmed, program loop rule].

---

# 8. Existing-repo map: what he already owns that seeds the build

Source: github-portfolio-refinement-plan.md, live inventory of 61 repos pulled 2026-10-18 [confirmed]. Only three true seeds exist. This section names them, then names the rest of the rescale stock honestly as proof shelf, not seed.

| Roadmap project | Seed repo on the account today | State of the seed |
| --- | --- | --- |
| P17 signal-ml-lab | drawing-modality-test (PulseKey, his drawing analysis research system, live demo) | Bucket A, rescale. Rename the public face to PulseKey in the description; add the Learn path. Its duplicate, DrawingModalitySprint3, archives (Batch 1) |
| P18 voice-agent-lab | SonicClear-AI-Studio-Grade-Audio-Enhancer (his audio cleaning product) | Bucket A, rescale. Add before and after samples and one quality number; frame it as this project's Learn base |
| P08 orchestrator-lab | Azure-ai-foundry-erp-swarm (fork, zero commits by him) | Bucket C, archive. A reading reference on the topic, not a seed. It holds no work of his and this file claims none |

Experience seed, not a repo: P06 grows from his real RAG evaluation work (a pipeline pass rate from 72.7 percent to 95 percent or more) [self-reported]. No public repo holds it today. The build starts from his private working material and his memory of the system, and the repo is where the claim becomes checkable.

Proof shelf, no roadmap seed (bucket A rescale stock in the refinement plan): jesun-code (his language; the flagship proof that sits outside this roadmap by design), PRISM, NoPara, Jesun_Dev, BigFish, dingdong-bms, SecureStep. These repos prove range and shipping. They seed no roadmap project, and v2 does not force a mapping onto them. Their Batch 2 fixes (descriptions, results tables, pins) run on the refinement plan's own track.

Every other project (P01 to P05, P07, P09 to P16, P19 to P22) starts as a new repo. That is the honest count: 19 of 22 projects are new builds, 2 grow from owned seeds, 1 grows from owned experience.

---

# 9. Revised build order and the fit math

The frame stands [confirmed, the-whole-board-plan.md]: program window 19 October 2026 to 18 October 2028, about 104 weeks; one project per about 6 weeks; about 17 slots inside the window; gap closure at 41.7 items per month.

Slots 1 to 17 keep the board plan order, unchanged. Nothing inside the window moves.

| Slot | Project | Phase |
| --- | --- | --- |
| 1 | P06 rag-eval-harness | A |
| 2 | P01 llm-from-scratch | A |
| 3 | P05 rag-foundation | B |
| 4 | P07 agent-from-scratch | B |
| 5 | P03 vllm-serving-lab | C |
| 6 | P04 cuda-kernels-lab | C |
| 7 | P08 orchestrator-lab | C |
| 8 | P19 distributed-training-lab (proposed) | C |
| 9 | P09 mcp-toolkit | D |
| 10 | P13 guardrails-lab | D |
| 11 | P10 finetune-lab | D |
| 12 | P12 llm-observability | D |
| 13 | P11 quantization-lab | E |
| 14 | P02 tokenizer-lab | E |
| 15 | P16 document-intelligence | E |
| 16 | P14 structured-outputs-lab | E |
| 17 | P18 voice-agent-lab | E |
| 18 | P20 ml-platform-lab (NEW) | F, past the window |
| 19 | P15 local-first-ai | F, past the window |
| 20 | P17 signal-ml-lab | F, past the window |
| 21 | P21 systems-languages-lab (NEW) | F, past the window |
| 22 | P22 multimodal-lab (NEW) | F, past the window |

Phase F order, with the reason [plan choice]: P20 first, because it closes a top-5 career gap and a 50-item category, and the board plan had already placed P15 and P17 past the window, so P20 displaces no in-window work; it steps ahead of them on demand weight. P15 and P17 follow in their board plan order. P21 and P22 close the list: their demand is real, and their trend years (2027 to 2028) are the latest on the board.

The fit math, stated plainly:
- The full list is 22 projects. At 6 weeks each, the list needs 132 weeks [arithmetic].
- The window holds about 104 weeks. 17 projects fit. 5 projects (P20, P15, P17, P21, P22) land past month 24 [arithmetic].
- At the same pace, slot 22 ends about 28 weeks past the window, near the end of April 2029 [arithmetic].
- The board plan already accepted 2 projects past the window (P15, P17) and said so openly. v2 adds 3 more to that tail and says so the same way. No pace is compressed to hide the tail.

What the NEW projects displace: inside the window, nothing. The displacement falls on the tail: P15 moves from slot 18 to slot 19 and P17 from slot 19 to slot 20, because P20 steps ahead of both. If Jesun rules the platform lane his primary target lane, the honest in-window trade exists and this file names it: swap P20 into slot 17 and move P18 to the tail. Voice breadth is the cheapest in-window slot to give up, because P18's posting family (xAI Multimodal, Mistral audio) is narrower than the platform family [confirmed counts, Section B]. The swap is his call under board plan decision 17, not this file's.

Expansion load, stated so the math stays honest: the EXPAND tracks add work inside their slots. The JAX track in P01 and the GRPO track in P10 each add about a week of work inside a 6-week slot [unverified estimate, plan judgment]. The rule that absorbs them: each expanded project cuts its own breadth list first (fewer datasets, fewer model sizes), and the Deep finish is never the cut (board plan cut order, step 2). If a track cannot be absorbed, Phase F grows by that week; the window order still does not change.

Two sub-steps ride free inside other slots, by the register's own design [confirmed]: the P20 Learn drills begin inside P03 (container drill, item 621) and inside P19 (a training job on a small cluster, item 624). P20's slot is late; its first skills are not.

---

# 10. Self-audit

Method: this file was written, then reopened and checked by script and by read, per the standing delivery rule (audit first, fix, then deliver). Checks ran on the file as it stood before this section was appended; this section adds no project content.

## Every v1 project accounted for

| v1 number | v2 number | Name | Ruling |
| --- | --- | --- | --- |
| 1 | P01 | llm-from-scratch | EXPAND (JAX track) |
| 2 | P02 | tokenizer-lab | KEEP |
| 3 | P03 | vllm-serving-lab | EXPAND (reasoning serving, container drill) |
| 4 | P04 | cuda-kernels-lab | EXPAND (standalone C++ tool) |
| 5 | P05 | rag-foundation | KEEP |
| 6 | P06 | rag-eval-harness | EXPAND (agent evaluation track) |
| 7 | P07 | agent-from-scratch | KEEP |
| 8 | P08 | orchestrator-lab | KEEP |
| 9 | P09 | mcp-toolkit | KEEP |
| 10 | P10 | finetune-lab | EXPAND (reasoning and RL track) |
| 11 | P11 | quantization-lab | EXPAND (TensorFlow optimization track) |
| 12 | P12 | llm-observability | KEEP (Go service moves to P21) |
| 13 | P13 | guardrails-lab | EXPAND (red team, governance artefacts) |
| 14 | P14 | structured-outputs-lab | KEEP |
| 15 | P15 | local-first-ai | KEEP |
| 16 | P16 | document-intelligence | KEEP |
| 17 | P17 | signal-ml-lab | EXPAND (TensorFlow depth track) |
| 18 | P18 | voice-agent-lab | KEEP (scope narrowed to audio) |
| proposed in career doc | P19 | distributed-training-lab | PROPOSED, ruled IN by this plan (Section 3) |
| new | P20 | ml-platform-lab | NEW |
| new | P21 | systems-languages-lab | NEW |
| new | P22 | multimodal-lab | NEW |

Count check: 18 v1 projects ruled (8 EXPAND, 10 KEEP, 0 MERGE, 0 DROP), plus P19, plus 3 NEW, equals 22 [arithmetic].

## Checks run, with results

1. Numbering: project headings run P01 to P22 with no gap and no repeat (script count: 22 headings) [confirmed, scan this run].
2. Shape: every NEW project (P20, P21, P22) and P19 carries full Learn, Build, Deep text in this file. Every EXPAND project names its added track in Learn, Build, Deep form. Every KEEP project states that the v1 level text carries over [confirmed, read this run].
3. Mapping: the Section 6 table holds 22 rows, one per project, each with job evidence, register categories, CEO atoms, and trend years (script count of table rows that start with a project code: 22 in the mapping table, 25 in the file including the 3 seed rows in Section 8) [confirmed, scan this run].
4. Fan-out: 22 fan-out lines, one per project, each naming a book chapter, a paper angle, and a course module. One honest exception is named in its own line: P21 publishes an essay, not a paper [confirmed, scan this run].
5. Fit math: 22 projects at 6 weeks equals 132 weeks; the window holds about 104; 17 slots inside, 5 past (slots 18 to 22); the tail ends about 28 weeks past the window [arithmetic, rechecked].
6. Slot order: slots 1 to 17 match the board plan's phase lists exactly (Phase 1: 6, 1, 5, 7; Phase 2 order inside C and D: 3, 4, 8, 19, 9; Phase 3: 13, 10, 12, 11; Phase 4 order inside E and F: 2, 16, 14, 18, 15, 17) [confirmed against the-whole-board-plan.md Section 2].
7. Counts quoted from sources were rechecked against the files: Section B counts (27, 31, 33, 22, 12, 10, 8, 4, 3 of 56), register category sizes (80, 70, 65, 60, 55, 55, 50, 50, 50, 45, 45, 40, 40, 35, 35, 35, 35, 20), the cross-match census (HAS 8, PARTIAL 12, MISSING 45 of 65; scale atoms 0 of 19), and the refinement plan buckets (61 repos) [confirmed].
8. Register closer counts per project were counted by script in this run (P19 at 59 mentions in Category 10 closers; P01 at 46 in Category 3; P03 at 37 in Category 9; P06 at 34 in Category 14; P10 at 33 in Category 8; P13 at 30 in Category 15; P04 at 39 in Category 11) [confirmed, script output].
9. Trend claims: every forward claim in Section 7 carries a source URL and a tag. Four browser searches ran (agents, inference and test-time compute, evaluations and safety governance, world models and AI infrastructure), inside the limit of four. No trend claim stands without a citation [confirmed, read this run].
10. Character scan: em dash count 0 and en dash count 0 in the file body before this section [confirmed, character scan this run]. Ranges use words; hyphens appear only inside names and numbers.
11. Scope: roadmap v1 was not edited. No other existing file was edited. GitHub stayed read-only: no repo was created, edited, archived, or deleted in this build [confirmed].

What this audit did not do: it did not re-read the 57 postings (the career documentation is the record of that read, and the loop re-grades it on its own cycle), and it did not open the primary text of the Gartner, Goldman, Morgan Stanley, or GlobalData reports. Their figures stay tagged [unverified] wherever this file uses them.

---

# Update (2026-10-19, evening): P19 hardware state, Spark boundary, live repo count, Jesun_Dev twin

Appended under the knowledge base update order of 2026-10-19. The body above stands as written on its build date; this note carries the facts that moved after it. No register count and no score in this file changes here; those move only through the program loop's gated cycles.

- P19 hardware, learning side [confirmed, live check 2026-10-19]: the BlackBox stack is installed and CUDA-proven. Python 3.14.7 with torch 2.11.0+cu128 on the NVIDIA GeForce RTX 5050 Laptop GPU (the cu128 pin fits driver 577.05); a 2048x2048 CUDA matmul ran clean. Record: ~/workspace/blackbox/p19-stack/status-2026-10-19.md. P19 Learn and Build steps now run on BlackBox. The publishable proof stays one rented cloud 2 GPU run under the CAD 50 gate, on Jesun's explicit call only (board plan FREE-ONLY tightening, 2026-10-19).
- Spark boundary [confirmed, 2026-10-19]: both Gallea DGX Sparks (spark-6aa3 and spark-31c4) are reachable from the Muse VM and inventoried (~/workspace/spark-gateway/). They are org machines under Jesun's R&D testing approval. No roadmap project routes personal work or a personal proof to a Spark. The P19 route stays BlackBox plus rented cloud, exactly as the Section 3 ruling sets it.
- Live repo count [confirmed, GitHub, 2026-10-19]: 44 public repos in total (37 authored, 7 forks); 54 owned in total with the 10 private. The 61-repo figures quoted in this file are the refinement plan's 2026-10-18 snapshot and stay as build history; 44 is the live count. The graded statement stands at 418 of 1000 on that count (the-statement-expanded-and-graded.md, Cycle 39 restatement, Pack 2026-10-19-06).
- Jesun_Dev twin [confirmed, 2026-10-19]: Section 8 names Jesun_Dev on the proof shelf. Its "AI Twin (Gemini 2.5)" claim was false as shipped (the shipped twin.js is a rule-based matcher). The fix package is built and mock-tested 18 of 18: a Cloudflare Worker proxy holds the key, and the site falls back to the matcher on any error. Deploy waits on a Gemini key from Jesun's Google AI Studio (his sign-in). The Batch 2 description, topics, and README for Jesun_Dev stay held until the live twin verifies. Package: the career goal's hidden_files/twin-build/.

Changelog: 2026-10-19, this update appended (P19 BlackBox stack state and proof route, Spark org-only boundary, live repo count 44 and statement 418 pointer, Jesun_Dev twin status). No body text changed.

## Acceptance criteria (scaffold level)

- Learn closes when the level-1-learn code runs, its README teaches the idea, and a fresh-eyes reader can follow it.
- Build closes when the system runs end to end, tests pass, and one benchmark or result table with method is in the README.
- Deep closes when the Deep scope in the section text is done, the failure modes are written, and the dossier is updated to ship state.
- No level closes on a claim. Evidence only: code, tests, numbers, links.
