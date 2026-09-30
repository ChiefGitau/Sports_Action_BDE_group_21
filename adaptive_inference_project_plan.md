# Adaptive Inference at the Edge — Project Plan

**Dataset:** SoccerTrack v2 (positional data only: `gsr/` + `bas/`, no video)
**Task:** Classify short windows of player positions as *action* vs. *no action*
**Approach:** Early-exit model on a shared trunk + knowledge distillation + INT8 quantization, with exit thresholds driven by a budget signal (e.g. battery level)

---

## 1. Architecture overview

One central **Core** module defines all shared formats and interfaces. Every other module only reads and writes those formats, so people can work in parallel from day one, using mock data until the real pieces arrive.

```
                 ┌────────────── M0 Core ──────────────┐
                 │ config · data contract · model API   │
                 │ FLOPs util · metrics · results.csv   │
                 └──────────────────────────────────────┘
                     ▲ everything imports from Core
                     │
 M1 Data ──► processed tensors ──► M2 Baselines ──────────────┐
                               └─► M3 Early-exit + KD ──┐     │
                                                       ▼     ▼
                         M4 Profiling (FLOPs table)   saved logits
                                   │                     │
                                   ▼                     ▼
                              M5 Exit policy ──► thresholds.json
                                   │
 M6 Compression (INT8/pruning) ────┤
                                   ▼
                   M7 Evaluation & Pareto ──► M8 Presentation
```

**Key design rule:** modules communicate through saved files with a fixed format, not through each other's code.

---

## 2. Modules

### M0 — Core (central module, built first, 2–3 days)

One person owns it. It sets the template everyone else follows.

- [ ] **Repo layout:** `core/`, `data/`, `models/`, `policy/`, `compression/`, `eval/`, `configs/`, `outputs/`
- [ ] **Config:** one YAML file for window length, splits, number of exits, seeds and paths. No hardcoded values in other modules.
- [ ] **Data contract:**
  - `X`: float32 `[N, T, P, 2]` (windows × frames × people × x,y)
  - `mask`: bool `[N, T, P]` (which people are present)
  - `y`: int64 `[N]` (0 = no action, 1 = action)
  - `meta`: match id, half, start frame per window
  - Saved as `data/processed/{train,val,test}.pt`
- [ ] **Mock data generator** producing random tensors in exactly that format
- [ ] **Model interface** every model implements:
  - `forward(x, mask) -> list[Tensor[N, C]]` — one entry per exit (a static baseline returns a list of one)
  - `num_exits`
  - `forward_until(x, mask, exit_idx)` — for per-exit FLOPs measurement
- [ ] **Shared utilities:** FLOPs measurement (one tool, e.g. fvcore), metrics (accuracy, balanced accuracy, F1), fixed seeds
- [ ] **Output formats:**
  - `logits_{split}.pt`: shape `[E, N, C]` plus labels
  - `flops.json`: FLOPs per exit
  - `thresholds.json`: `{budget_level: [threshold per exit]}`
  - `results.csv`: one row per operating point — `model, variant, budget, split, flops_mean, acc, bal_acc, f1, size_mb, latency_ms`

**Done when:** mock data loads, a dummy model runs through the interface, and one row lands in `results.csv`.

---

### M1 — Data pipeline

- [ ] Request access on Hugging Face; download only `gsr/` and `bas/`
- [ ] Parse one half-match; check frame rate, JSON structure, people per frame
- [ ] Windowing and labelling per config (start with 0.5 s windows)
  - Positive if an event frame falls inside the window
  - Drop or add tolerance for windows with an event right at the edge
- [ ] **Do not use the ball track as input** — it's interpolated between events, so its kinks leak the labels
- [ ] Fix person ordering (e.g. by team, then role) and padding/masking
- [ ] Check class balance at 0.5 s and 1 s; pick the window length
- [ ] Sanity-check a few windows by plotting positions around a known event
- [ ] Write processed tensors per official split:
  - train: M1, M3, M4, M5, M6, M8
  - validation: M2, M10
  - test: M7, M9
- [ ] Short data report (sizes, class balance, window length choice)

**Input:** raw dataset · **Output:** `data/processed/*.pt`
**Independent because:** others use mock data until the integration checkpoint.

---

### M2 — Static baselines

- [ ] Full model with no exits (1D-CNN or small transformer)
- [ ] Statically pruned or half-width version — **use physical removal** (e.g. Torch-Pruning), not masking, or FLOPs won't drop
- [ ] Train both, record metrics and FLOPs, then **freeze them** — nobody changes baselines after this

**Input:** processed tensors · **Output:** checkpoints + rows in `results.csv`

---

### M3 — Early-exit model and distillation

- [ ] Shared trunk with 3–4 exit heads, implementing the Core model interface
- [ ] Keep exit heads lightweight (pooling + one linear layer)
- [ ] Joint training with weighted per-exit losses
- [ ] Knowledge distillation: early exits learn from the final head's soft outputs
- [ ] Check each head's validation accuracy; move exits if the earliest is near random
- [ ] Dump logits for every exit on validation and test

**Input:** processed tensors · **Output:** checkpoint + `logits_{val,test}.pt`

---

### M4 — Cost profiling

- [ ] FLOPs per exit (including the exit head and confidence check), parameter count, size on disk
- [ ] Optional: CPU latency at **batch size 1** (matches the streaming scenario)
- [ ] Verify weight sharing: compare trunk weights layer by layer to show one deployable artifact

**Input:** any checkpoint · **Output:** `flops.json` + profiling report
**Independent because:** can be built and tested against Core's dummy model.

---

### M5 — Exit policy and budget controller

- [ ] Confidence measures: max softmax, entropy (optionally margin)
- [ ] Calibration check; temperature scaling if needed
- [ ] Controller mapping budget level (e.g. battery 100% / 50% / 20%) to thresholds
- [ ] Tune thresholds on **validation only**
- [ ] Re-tune thresholds for the INT8 model (calibration shifts after quantization)
- [ ] Runtime function: given one window and a budget → prediction + exit used

**Input:** `logits_val.pt`, `flops.json` · **Output:** `thresholds.json` + policy code
**Independent because:** works only on saved logits; fake logits suffice until M3 delivers.

---

### M6 — Compression

- [ ] Post-training INT8 quantization of the early-exit model
- [ ] Calibration set must include *action* windows, not just mostly *no action*
- [ ] Accuracy and size before and after
- [ ] Optional: structured pruning of the trunk

**Input:** checkpoints · **Output:** compressed checkpoints + `results.csv` rows

---

### M7 — Evaluation and Pareto

- [ ] Test-set operating points: apply `thresholds.json` to `logits_test.pt`, weight by `flops.json`
- [ ] Pareto plot: accuracy (and balanced accuracy/F1) vs. mean FLOPs, with both baselines, float and INT8
- [ ] Analysis: which kinds of windows exit early vs. go deep
- [ ] Runtime demo: simulated battery draining over a match half, compute per window dropping
- [ ] Honest comparison: does adaptive beat static pruning at matched FLOPs? If not, explain why

**Input:** `results.csv` + saved files · **Output:** figures + analysis
**Independent because:** can build all plots from mock `results.csv` rows from day one.

---

### M8 — Presentation

- [ ] Scenario and motivation: edge camera box at an amateur club processing the tracking stream
- [ ] Design and toolbox choices (why early exit, why KD, why INT8)
- [ ] Pareto plot and interpretation
- [ ] Trade-offs, what didn't work, novelty
- [ ] Collect "what didn't work" notes from every module owner throughout, not only at the end
- [ ] Clean up code so the Pareto frontier reproduces from saved tensors

---

## 3. Schedule

| Phase | When | What |
|---|---|---|
| 0 | Week 4, days 1–3 | M0 Core by one person; others read the dataset docs and assigned readings, and plan their module |
| 1 | Week 4 → mid week 5 | M1–M7 in parallel on mock data |
| Integration checkpoint | End of week 5 | Real tensors from M1 through M2 and M3; swap mocks for real files; fix format mismatches |
| 2 | Week 6 | Real runs: M5 tunes thresholds, M6 compresses, M7 produces the real Pareto plot |
| 3 | Week 7 | Analysis, runtime demo, M8 slides, presentation |

---

## 4. Dividing the work

**3 people**
1. M0 Core, then M1 Data
2. M2 Baselines + M3 Early-exit
3. M4 Profiling + M5 Policy + M7 Evaluation

M6 and M8 are shared.

**4 people:** split person 3 — one takes M5 + M6, the other M4 + M7.

---

## 5. Rules for working in parallel

- Only the Core owner changes Core formats; any change is announced to the group
- Each module has a small test that runs on mock data
- Every experiment writes to `results.csv` through the Core helper — no custom formats
- One branch per module, merged via short reviews

---

## 6. Background readings per module

| Module | Reading | Slides that matter most |
|---|---|---|
| **M0 Core** | week4_adaptivity | 13–14 (adaptivity loop, four design axes), 25 (anytime vs. budgeted inference) |
| **M1 Data** | Week1_Intro to course | 28–32 (event-driven/bursty data), 43 (resampling and windowing), 48 (validation) |
| **M2 Baselines** | pruning_lecture_2 + DepGraph paper (2301.12900) | 15–24 (structured vs. unstructured), 48 (masking vs. physical removal), 50 (layer sensitivity); Torch-Pruning library |
| **M3 Early-exit + KD** | week4_adaptivity, Final lecture | Adaptivity 35–43 (exit heads, loss weighting, failure modes); Final lecture 4–18 (distillation loss, temperature, self-distillation) |
| **M4 Profiling** | pruning_lecture_2, week1_edgeAI_intro | Pruning 54–61 (evaluation metrics); Edge intro 36, 55 (profile first, evaluate full system), 25–32 (ONNX, if measuring latency) |
| **M5 Exit policy** | week4_adaptivity | 28–29 (policy signals, calibration), 41 (exit criteria), 43 (premature exits, imbalance, drift) |
| **M6 Compression** | Quantization_2 | 36–39 (PTQ pipeline, calibration data), 29 (per-layer schemes), 43 (PTQ vs. QAT), 44 (deployment checklist) |
| **M7 Evaluation** | week4_adaptivity | 15 (Pareto front), 30 (accuracy is not enough), 32–34 (when adaptivity helps, pitfalls) |
| **M8 Presentation** | Final lecture, Week1_Intro to course | Final 66 (design decision map); Week1 53–68 (edge constraints) |

**Less useful for this project:** Quantization_1 (bit-level number formats), FlexInt paper (custom sub-8-bit format, not runnable in PyTorch), adaptivity slides 44–68 (slimmable/Elastoformer — only slide 51 is worth citing to justify early exit), Final lecture 27–64 (LoRA, NAS, federated learning; slides 52–56 on drift could be a small novelty angle).

**Suggested reading per person (3-person split)**
- **Person 1:** Week1 intro 28–48; adaptivity 13–14, 25
- **Person 2:** adaptivity 35–43; Final lecture 4–18; pruning 15–24, 48; DepGraph sections 1–3
- **Person 3:** adaptivity 15, 28–34, 41–43; pruning 54–61; Quantization_2 36–44

---

## 7. Warnings from the lectures

- **Masked pruning doesn't reduce counted FLOPs** — use physical removal for the pruned baseline
- **Calibration shifts after quantization** — tune exit thresholds per precision
- **Quantization calibration data needs rare cases** — include *action* windows
- **Unfair baselines and controller overhead** — count exit-head and policy cost; train baselines properly
- **Exit heads must stay light** — or they cancel the savings
- **Batching complicates early exit** — measure latency at batch size 1

---

## 8. Risks

- **Early heads too weak:** frontier gains little → strengthen early trunk layers or move exits
- **Class imbalance (~1/6 positive at 0.5 s):** report balanced accuracy and F1, not only accuracy
- **Label noise at window edges:** use a small tolerance or drop edge windows
- **Ball-track leakage:** never use the ball track as input
