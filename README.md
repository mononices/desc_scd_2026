# UAE Privacy Mesh

Privacy-preserving analytics on sensitive government data.

Three emirate health authorities jointly train a heart-risk model on their own synthetic
screening records — they share parameter updates, not raw data, and protect the updates
with DP-SGD under a per-entity budget ε. A membership-inference attack then measures how
much each configuration actually leaks.

## Reference results

Measured with 3 independent seeds (5 for ε=2); δ = 1e-5; attack pool of ~1,900 candidates.

| configuration | ε achieved | test accuracy | attack AUC |
|---|---|---|---|
| Centralised (raw data pooled) | ∞ | 0.731 ± 0.014 | 0.631 ± 0.004 |
| Federated (FedAvg, no DP) | ∞ | 0.762 ± 0.006 | 0.590 ± 0.008 |
| Federated + DP-SGD, ε = 8 | 7.4 | 0.772 ± 0.006 | 0.542 ± 0.014 |
| Federated + DP-SGD, ε = 4 | 3.7 | 0.782 ± 0.007 | 0.520 ± 0.006 |
| Federated + DP-SGD, ε = 2 | 1.9 | 0.765 ± 0.020 | 0.509 ± 0.025 |
| Federated + DP-SGD, ε = 1 | 0.9 | 0.658 ± 0.008 | 0.508 ± 0.025 |

Attack AUC 0.5 = chance. The naive centralised model leaks membership clearly; plain
federated averaging reduces the signal; DP-SGD drives it to chance level. Accuracy is
unchanged within noise until ε = 2 and drops ~10 points at ε = 1.

## Quick start

Docker (recommended):

```bash
docker compose up --build
```

Open http://localhost:7860. First boot pre-runs the full grid (3 repeats of 6
configurations, several minutes on CPU). Remove `results/` to regenerate the cache.

Local (Python 3.12):

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0
pip install -r requirements.txt
python -m privacy_mesh.cli prewarm   # run grid, cache to results/
python -m privacy_mesh.cli serve     # Gradio dashboard on :7860
```

## Layout

| path | role |
|---|---|
| `privacy_mesh/data.py` | synthetic populations and entity partitions |
| `privacy_mesh/federation.py` | FedAvg with per-entity DP-SGD |
| `privacy_mesh/attack.py` | membership-inference attack |
| `privacy_mesh/demo.py`, `app.py` | Gradio dashboard and entry point |
| `Dockerfile`, `compose.yaml` | container |
| `data/` | vendored UCI heart-disease cohort |

## How it works

- Data: synthetic citizens calibrated on the public UCI heart-disease cohort, split across
  three entities with slightly different populations plus a federal pool that never enters
  training and serves as the attacker's background.
- Federation: 12 rounds, 6 local epochs each; the aggregator averages the returned model
  states weighted by dataset size. Only model parameters cross the boundary.
- Privacy: Opacus DP-SGD per local step — per-sample gradient clipping (C = 1), Gaussian
  noise with Poisson sampling. σ is calibrated so the Rényi accountant reports ε ≤ target
  per entity over all rounds (δ = 1e-5).
- Attack: given the final model, a logistic regression on confidence features decides
  membership for 800 true members vs 1,100 non-members from the same populations,
  reported as ROC AUC.
- Everything is seeded; each configuration always reproduces the same numbers.

## Known limitations

- Results are specific to one tabular data regime. The observed leakage signal is modest
  (AUC 0.63 vs 0.5) even for a fully memorizing model, because most non-members are also
  predicted confidently; this is the normal state of affairs for score-based membership
  inference on tabular data, and the comparison across configurations is what matters.
- The ε guarantee is the Rényi-DP accounting implemented by Opacus 1.5 under Poisson
  sampling, computed per entity. The demo uses non-secure RNG (`secure_mode` off) for
  reproducibility; production deployments should re-train with secure mode enabled.
- The attack is an evaluation harness with known ground truth; in the field an attacker
  would have partial membership knowledge at best.
