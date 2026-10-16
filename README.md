# Sports Action Recognition with Adaptive Inference

Big Data at the Edge — Group 21

This project studies efficient sports-action recognition from SoccerTrack v2 tracking data. The goal is to compare a standard CNN, structured pruning, an early-exit neural network, adaptive confidence-based inference, and post-training INT8 quantization.

The final system evaluates the trade-off between predictive performance and computational cost using accuracy, balanced accuracy, F1 score, FLOPs, model size, and early-exit behavior.

## Quick start: run the demo

The interactive demo visualises the trained models on a real SoccerTrack clip. It needs Python 3.12 and the checkpoints already in `outputs/`; nothing is trained. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run notebooks/streamlit_app.py
```

The app opens at `http://localhost:8501`. Pick a model, a clip and a battery scenario in the sidebar, scrub through the frames or press **Render video** for the full mp4. Details in section 3 (setup) and section 11 (runtime demo).

---

## 1. Project pipeline

The project is organized into the following stages:

| Stage | Description |
|---|---|
| M1 | Download and preprocess SoccerTrack v2 data |
| M2 | Train full CNN baseline and physically pruned CNN |
| M3 | Train multi-exit CNN with knowledge distillation |
| M4 | Profile FLOPs, parameters, model size, and CPU latency |
| M5 | Tune confidence thresholds for adaptive early exiting |
| M6 | Apply post-training INT8 quantization |
| M7 | Perform final test-set evaluation and Pareto analysis |

The final evaluation compares:

- full FP32 CNN;
- structured-pruned FP32 CNN;
- full-depth FP32 early-exit CNN;
- adaptive FP32 early-exit inference;
- full-depth INT8 early-exit CNN;
- adaptive INT8 early-exit inference.

---

## 2. Repository structure

```text
Sports_Action_BDE_group_21/
|
|-- compression/
|   `-- quantization.py
|
|-- configs/
|   `-- default.yaml
|
|-- core/
|   |-- artifacts.py
|   |-- metrics.py
|   |-- policy.py
|   `-- ...
|
|-- models/
|   |-- baseline.py
|   |-- cnn.py
|   |-- early_exit.py
|   |-- prune.py
|   |-- registry.py
|   `-- train.py
|
|-- scripts/
|   |-- process_match.py
|   |-- build_dataset.py
|   |-- build_splits.py
|   |-- train_baseline.py
|   |-- prune_baseline.py
|   |-- train_early_exit.py
|   |-- profile_costs.py
|   |-- tune_policy.py
|   |-- run_quantization.py
|   `-- evaluate_final.py
|
|-- tests/
|
|-- outputs/
|   |-- cnn_full/
|   |-- cnn_pruned/
|   |-- earlyexit/
|   |-- earlyexit_int8/
|   `-- final/
|
|-- requirements.txt
`-- README.md
```

Canonical trained-model and final-result artifacts under `outputs/` are included in the repository. Other experimental outputs remain ignored by Git.

Raw SoccerTrack data and generated processed datasets are not committed.

---

## 3. Environment setup

The project has been tested with Python 3.12.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Verify the installation with:

```bash
python -m pytest
```

The current project test suite contains 28 tests covering the core utilities, models, adaptive policy, and quantization pipeline.

### Running the Streamlit demo

`notebooks/streamlit_app.py` is an interactive visualisation of the trained models: pick a SoccerTrack clip (or upload a video), a model (full, pruned, early-exit, INT8) and a battery scenario, and it renders the same kind of clip as `scripts/render_demo_clips.py` with the exits, predictions and compute per window. It only reads the checkpoints in `outputs/` and the processed splits in `data/processed/`; nothing is trained or re-tuned.

Streamlit is included in `requirements.txt`. With the virtual environment activated, run from the repository root (not from inside `notebooks/`, the app resolves its paths relative to the repo):

```bash
streamlit run notebooks/streamlit_app.py
```

It prints a local URL (normally `http://localhost:8501`) and opens it in the browser. Stop it with `Ctrl+C`. The clip picker lists every `.mp4` in `data/raw/frames/` (the clips written by `scripts/fetch_demo_frames.py`, see section 11); the included `117093_2nd_53077_240.mp4` works out of the box, and any other video can be uploaded from the sidebar.

---

## 4. Dataset

The project uses SoccerTrack v2 tracking and action-event data.

Each processed example follows the common data contract:

```text
X    : float32 [N, T, P, 2]
mask : bool    [N, T, P]
y    : int64   [N]
```

where:

- `N` is the number of windows;
- `T = 12` frames for a 0.5-second window at 25 FPS;
- `P = 23` person slots;
- the final dimension contains `(x, y)` coordinates;
- label `0` means no action;
- label `1` means action.

The data are split by complete matches rather than random windows to avoid train/test leakage.

### Canonical match split

Training matches:

```text
117092
118575
118576
118577
118578
128058
```

Validation matches:

```text
117093
132877
```

Test matches:

```text
128057
132831
```

The resulting canonical dataset contains:

```text
TRAIN: 70,722 windows
VAL:   22,927 windows
TEST:  22,841 windows
```

The training split contains 12,332 positive action windows. The validation split contains 4,468 positive windows, and the test split contains 4,319 positive windows.

Because the dataset is imbalanced, balanced accuracy and F1 are used alongside ordinary accuracy.

---

## 5. Rebuilding M1 data

The raw and processed data are intentionally excluded from Git because of their size.

`process_match.py` downloads and preprocesses one complete SoccerTrack v2 match.

For example:

```bash
python scripts/process_match.py --match-id 117092
```

To process all ten canonical matches in Windows PowerShell:

```powershell
$matches = @(
    "117092",
    "117093",
    "118575",
    "118576",
    "118577",
    "118578",
    "128057",
    "128058",
    "132831",
    "132877"
)

foreach ($match in $matches) {
    python scripts\process_match.py --match-id $match
}
```

After all matches have been processed, build the canonical train/validation/test splits with:

```bash
python scripts/build_splits.py
```

This creates:

```text
data/processed/train.pt
data/processed/val.pt
data/processed/test.pt
```

The split builder keeps complete matches separated between train, validation, and test to avoid leakage.

---

## 6. M2 — Static CNN baselines

### Full CNN

Train and export the full static CNN baseline:

```bash
python scripts/train_baseline.py
```

Canonical artifacts are written to:

```text
outputs/cnn_full/
```

including:

```text
model.pt
flops.json
logits_val.pt
logits_test.pt
```

The canonical full CNN contains 54,530 parameters.

Its computational cost is:

```text
1,285,888 FLOPs per sample
```

### Structured-pruned CNN

After training the full model, run:

```bash
python scripts/prune_baseline.py
```

The default pruning ratio is:

```text
0.5
```

This physically reduces the CNN trunk width from 64 channels to 32 channels.

The resulting pruned model contains 14,978 parameters and requires:

```text
348,032 FLOPs per sample
```

Artifacts are written to:

```text
outputs/cnn_pruned/
```

---

## 7. M3 — Early-exit network

Train the four-exit model with joint exit losses and knowledge distillation:

```bash
python scripts/train_early_exit.py
```

The architecture places exits after CNN blocks:

```text
1, 2, 3, 4
```

The training configuration uses:

```text
KD enabled:             true
KD alpha:               0.5
KD temperature:         2.0
exit loss weights:      [0.25, 0.25, 0.25, 1.0]
```

The final canonical early-exit model was selected using validation balanced accuracy.

Artifacts are written to:

```text
outputs/earlyexit/
```

The early-exit network contains 54,920 parameters.

The measured FLOPs per exit are:

```text
Exit 1:   401,152
Exit 2:   696,320
Exit 3:   991,488
Exit 4: 1,286,656
```

This allows inference to terminate before executing the complete network when confidence is sufficiently high.

---

## 8. M4 — Computational-cost profiling

Profile the trained early-exit model on CPU with:

```bash
python -m scripts.profile_costs --checkpoint outputs/earlyexit/model.pt --split data/processed/val.pt --warmup 10 --repeats 50
```

The profiler measures:

- parameter count;
- serialized model size;
- FLOPs per exit;
- batch-size-1 CPU inference latency.

The canonical profiling run measured approximately:

```text
Exit 1: 0.385 ms
Exit 2: 0.497 ms
Exit 3: 0.658 ms
Exit 4: 0.852 ms
```

The corresponding FLOPs were:

```text
Exit 1:   401,152
Exit 2:   696,320
Exit 3:   991,488
Exit 4: 1,286,656
```

Latency values are hardware-dependent and should be interpreted as measurements from the machine used for the experiment.

---

## 9. M5 — Adaptive early-exit policy

The adaptive policy uses maximum softmax probability as its confidence score.

Thresholds are tuned on the validation set only.

### FP32 policy

Run:

```bash
python -m scripts.tune_policy --logits outputs/earlyexit/logits_val.pt --flops outputs/earlyexit/flops.json --output outputs/earlyexit/thresholds.json --report outputs/earlyexit/policy_report.txt --method max_softmax --temperature 1.0 --grid-step 0.01
```

The validation-selected FP32 thresholds for the 100% and 50% operating points are:

```text
[0.52, 0.52, 0.52, 0.0]
```

The final exit always acts as the fallback.

For the validation split, this policy used approximately:

```text
457,902 mean FLOPs
```

which is about:

```text
35.6% of full early-exit compute
```

### 20% budget limitation

The first exit alone requires:

```text
401,152 FLOPs
```

while the full early-exit model requires:

```text
1,286,656 FLOPs
```

Therefore the minimum possible early-exit compute is approximately:

```text
31.2% of full-model compute
```

A true 20% compute budget is therefore architecturally impossible for this network.

For the requested 20% budget, the policy reports the minimum-compute operating point instead of incorrectly claiming that the 20% target was achieved.

---

## 10. M6 — INT8 post-training quantization

Run post-training static INT8 quantization with:

```bash
python -m scripts.run_quantization --train data/processed/train.pt --eval data/processed/val.pt --checkpoint outputs/earlyexit/model.pt --output-dir outputs/earlyexit_int8 --results outputs/results.csv --max-calibration 256 --min-action 32 --batch-size 32 --backend onednn
```

The calibration subset intentionally includes action windows so the minority action class is represented.

The canonical calibration run used:

```text
Calibration windows:        256
Action calibration windows: 64
Backend:                    oneDNN
```

The measured model sizes were:

```text
FP32: 0.233905 MB
INT8: 0.080355 MB
```

This corresponds to approximately:

```text
65.65% model-size reduction
```

The predictive performance remained essentially unchanged after quantization.

The reloadable INT8 model is stored as:

```text
outputs/earlyexit_int8/earlyexit_int8.pt
```

The deployable TorchScript checkpoint is slightly larger on disk because it also stores graph/runtime information.

### Retuning the INT8 policy

Quantization changes confidence values slightly, so FP32 confidence thresholds are not reused.

Retune the INT8 policy on validation logits with:

```bash
python -m scripts.tune_policy --logits outputs/earlyexit_int8/logits_val.pt --flops outputs/earlyexit/flops.json --output outputs/earlyexit_int8/thresholds.json --report outputs/earlyexit_int8/policy_report.txt --method max_softmax --temperature 1.0 --grid-step 0.01
```

The validation-selected INT8 thresholds for the main operating points are:

```text
[0.53, 0.53, 0.53, 0.0]
```

compared with:

```text
[0.52, 0.52, 0.52, 0.0]
```

for FP32.

This demonstrates why policy thresholds should be retuned after quantization.

---

## 11. M7 — Final evaluation

M7 evaluates all canonical operating points on the held-out test set:

```bash
python -m scripts.evaluate_final
```

M7 does not tune thresholds using the test set.

The experimental procedure is:

```text
validation set -> select model / thresholds
test set       -> final evaluation only
```

The script also verifies that the labels stored with each set of logits match the canonical test labels before combining results.

This prevents models evaluated on incompatible dataset splits from being compared accidentally.

Final artifacts are written to:

```text
outputs/final/
```

including:

```text
results.csv
pareto_front.csv
policy_details.json
evaluation_report.txt
balanced_accuracy_vs_flops.png
f1_vs_flops.png
balanced_accuracy_vs_size.png
```

### Runtime demo

`scripts/demo_runtime.py` shows the early-exit model working on real windows, one at a time, with the trunk really stopping at the first confident head:

```bash
python -m scripts.demo_runtime trace   --split data/processed/val.pt --start 4423 --windows 10
python -m scripts.demo_runtime battery --split data/processed/val.pt --capacity-fraction 0.35
```

`trace` prints one row per window: the confidence at every head that ran, the exit taken, FLOPs used and single-thread CPU latency. `battery` streams a whole split while a simulated battery drains in proportion to FLOPs, switching to the 50% and 20% threshold sets as it crosses those levels. Any `.pt` in the data contract works, so one freshly processed match half is enough for a live demo. Nothing is re-tuned.

Two 10 s video clips (1280x720, 25 fps) of the same 20 windows, one per model, are rendered with:

```bash
python scripts/fetch_demo_frames.py     # once: 10 s of the real camera video, fetched by HTTP range requests
python scripts/render_demo_clips.py     # outputs/demo/clip_earlyexit.mp4, clip_fullcnn.mp4, still_*.png
```

The same visualisation is available interactively in the Streamlit app (`streamlit run notebooks/streamlit_app.py`, see section 3), where the model, clip and battery scenario can be changed and the clip rendered or scrubbed frame by frame.

Each clip shows the real SoccerTrack panorama frames with the tracked player boxes, the bird's-eye view of the same tracking data, which trunk blocks ran, the exit taken, and the cumulative compute. SoccerTrack publishes one 4096x1080 panorama video per half (3.5 GB each); `fetch_demo_frames.py` seeks into it over HTTPS and stores only the needed frames (a few MB), so the whole video is never downloaded. It needs a Hugging Face login that has accepted the dataset terms. Video frame `N` is annotation image `N`, which the script verifies by also extracting the image-space boxes for the same frames.

---

## 12. Final test results

All results below use the same canonical test set of:

```text
22,841 windows
```

| Model | Variant | Mean FLOPs | Accuracy | Balanced accuracy | F1 | Size MB |
|---|---|---:|---:|---:|---:|---:|
| CNN full | FP32 | 1,285,888 | 0.5202 | 0.5392 | 0.3099 | 0.2305 |
| CNN pruned | FP32 + pruning | 348,032 | 0.3801 | **0.5530** | **0.3365** | 0.0712 |
| Early-exit | FP32 final exit | 1,286,656 | 0.4270 | 0.5493 | 0.3299 | 0.2339 |
| Early-exit | INT8 final exit | 1,286,656 | 0.4258 | 0.5483 | 0.3293 | 0.0804 |
| Early-exit | FP32 adaptive | 472,046 | 0.4540 | 0.5483 | 0.3265 | 0.2339 |
| Early-exit | INT8 adaptive | 529,927 | 0.4442 | 0.5480 | 0.3273 | 0.0804 |

The values shown for the adaptive FP32 and INT8 models correspond to the main validation-selected policy operating points.

Because the dataset is imbalanced, balanced accuracy and F1 are more informative than ordinary accuracy alone.

### Full CNN

The full CNN achieves:

```text
Balanced accuracy: 0.5392
F1:                0.3099
FLOPs:             1,285,888
```

### Pruned CNN

The physically pruned CNN achieves:

```text
Balanced accuracy: 0.5530
F1:                0.3365
FLOPs:               348,032
```

Despite using much less computation than the full CNN, it achieves the strongest balanced accuracy and F1 among the tested operating points.

### FP32 adaptive early exit

The main FP32 adaptive policy achieves approximately:

```text
Mean FLOPs:        472,046
Balanced accuracy: 0.5483
F1:                0.3265
```

This is substantially less computation than executing the full early-exit network to exit 4.

### INT8 adaptive early exit

The main INT8 adaptive policy achieves approximately:

```text
Mean FLOPs:        529,927
Balanced accuracy: 0.5480
F1:                0.3273
Model size:        0.0804 MB
```

This combines dynamic early exiting with reduced model storage.

---

## 13. Pareto analysis

For the measured balanced-accuracy-versus-FLOPs trade-off, the structured-pruned CNN is the only non-dominated operating point.

Its result is:

```text
FLOPs:             348,032
Balanced accuracy: 0.5530
F1:                0.3365
```

This means that, among the operating points evaluated in this experiment, no other model simultaneously provides equal-or-better balanced accuracy at equal-or-lower computational cost.

This is an experimental result rather than an assumption.

In this particular experiment, structured pruning produced a stronger compute/quality trade-off than the adaptive early-exit alternatives.

The adaptive early-exit models still demonstrate that individual samples can terminate before the final network depth and substantially reduce average computation relative to full-depth inference.

---

## 14. Canonical artifacts

The repository contains the canonical trained-model and result artifacts required to inspect the reported experiments.

### Full CNN

```text
outputs/cnn_full/
```

Contains:

```text
model.pt
flops.json
logits_val.pt
logits_test.pt
```

### Pruned CNN

```text
outputs/cnn_pruned/
```

Contains:

```text
model.pt
flops.json
logits_val.pt
logits_test.pt
```

### FP32 early-exit model

```text
outputs/earlyexit/
```

Contains:

```text
model.pt
flops.json
logits_val.pt
logits_test.pt
thresholds.json
policy_report.txt
```

### INT8 early-exit model

```text
outputs/earlyexit_int8/
```

Contains:

```text
earlyexit_int8.pt
quantization_metadata.pt
quantization_report.txt
logits_val.pt
logits_test.pt
thresholds.json
policy_report.txt
```

### Final evaluation

```text
outputs/final/
```

Contains:

```text
results.csv
pareto_front.csv
policy_details.json
evaluation_report.txt
balanced_accuracy_vs_flops.png
f1_vs_flops.png
balanced_accuracy_vs_size.png
```

Non-canonical experimental runs remain ignored through `.gitignore`.

---

## 15. Running the test suite

Run:

```bash
python -m pytest
```

At the time of the final experiment:

```text
28 tests passed
```

The tests cover:

- artifact loading and validation;
- model interfaces;
- baseline CNN behavior;
- early-exit model behavior;
- physical pruning;
- confidence-policy behavior;
- calibration-data selection;
- real quantized modules;
- INT8 output shapes.

---

## 16. PyTorch warnings

Current PyTorch versions may print warnings indicating that:

```text
torch.ao.quantization
```

and TorchScript APIs are scheduled for future deprecation.

These warnings do not indicate a failed experiment.

The current project environment successfully:

- performs FX post-training INT8 quantization;
- creates real quantized modules;
- exports the INT8 model;
- reloads the exported checkpoint;
- executes inference;
- passes the complete test suite.

A future implementation could migrate the quantization pipeline to newer `torchao` / PT2E APIs.

---

## 17. Full reproduction order

For a completely fresh reproduction:

```text
1. Create a Python environment
2. Install requirements
3. Process all 10 SoccerTrack matches
4. Build train/validation/test splits
5. Train cnn_full
6. Prune and fine-tune cnn_pruned
7. Train the early-exit model
8. Profile computational cost
9. Tune FP32 thresholds on validation
10. Run INT8 post-training quantization
11. Tune INT8 thresholds on validation
12. Run final M7 evaluation on test
13. Run pytest
```

The main commands are:

```bash
python scripts/build_splits.py

python scripts/train_baseline.py

python scripts/prune_baseline.py

python scripts/train_early_exit.py

python -m scripts.profile_costs --checkpoint outputs/earlyexit/model.pt --split data/processed/val.pt --warmup 10 --repeats 50

python -m scripts.tune_policy --logits outputs/earlyexit/logits_val.pt --flops outputs/earlyexit/flops.json --output outputs/earlyexit/thresholds.json --report outputs/earlyexit/policy_report.txt --method max_softmax --temperature 1.0 --grid-step 0.01

python -m scripts.run_quantization --train data/processed/train.pt --eval data/processed/val.pt --checkpoint outputs/earlyexit/model.pt --output-dir outputs/earlyexit_int8 --results outputs/results.csv --max-calibration 256 --min-action 32 --batch-size 32 --backend onednn

python -m scripts.tune_policy --logits outputs/earlyexit_int8/logits_val.pt --flops outputs/earlyexit/flops.json --output outputs/earlyexit_int8/thresholds.json --report outputs/earlyexit_int8/policy_report.txt --method max_softmax --temperature 1.0 --grid-step 0.01

python -m scripts.evaluate_final

python -m pytest
```

---

## 18. Configuration

The shared experiment configuration is stored in:

```text
configs/default.yaml
```

This file defines:

- random seed;
- dataset paths;
- window and stride duration;
- maximum number of people;
- number of classes;
- CNN architecture;
- early-exit positions;
- training parameters;
- knowledge-distillation parameters;
- structured-pruning ratio;
- battery-budget levels.

Keeping these values in a shared configuration file avoids inconsistent hard-coded settings between experiment stages.

---

## 19. Notes on reproducibility

The canonical model artifacts and final result files are included in the repository so the reported results can be inspected without retraining every neural network.

The raw SoccerTrack dataset and generated `data/processed/*.pt` files are excluded from Git.

Therefore:

- the included output artifacts can be inspected directly after cloning;
- running `python -m scripts.evaluate_final` requires the canonical processed test split to be present;
- a completely fresh reproduction should first rebuild M1 data as described above.

Model selection and threshold tuning are performed on validation data only. The held-out test split is reserved for final evaluation.
