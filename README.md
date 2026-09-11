# UAE Privacy Mesh

Privacy-preserving analytics on sensitive government data.

Three emirate health authorities jointly train a heart-risk model on synthetic screening
records. They share parameter updates, not raw data, and protect those updates with
DP-SGD under a stated per-entity budget ε. A membership-inference attack then measures
how much each configuration leaks.

## Quickstart

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate           # macOS / Linux
# or: .venv\Scripts\Activate.ps1    # Windows (PowerShell)
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements.txt
```

No GPU is required.

```bash
python -m pytest tests/ -q          # run property tests
python -m privacy_mesh.cli prewarm --force   # precompute all experiments
python -m privacy_mesh.cli serve             # launch Gradio dashboard
```

Open http://localhost:7860. Docker: `docker compose up --build`.

## Reference results

Three independent seeds per configuration, δ = 1e-5, balanced attack pool of 3,000
candidates (1,500 members vs 1,500 distribution-matched non-members).

| configuration | ε target | ε spent | test acc | train acc | gap | attack acc | attack AUC |
|---|---|---|---|---|---|---|---|
| Centralised (raw data pooled) | - | ∞ | 0.728 ±0.012 | 1.000 | +0.272 | **70.2%** | 0.672 ±0.004 |
| Federated (FedAvg, no DP) | - | ∞ | 0.738 ±0.012 | 0.871 | +0.133 | 55.5% | 0.565 ±0.010 |
| Federated + DP-SGD | 8 | 7.41 | **0.785 ±0.002** | 0.790 | +0.005 | 50.3% | 0.505 ±0.008 |
| Federated + DP-SGD | 4 | 3.70 | 0.777 ±0.012 | 0.782 | +0.005 | 49.8% | 0.492 ±0.004 |
| Federated + DP-SGD | 2 | 1.85 | 0.738 ±0.030 | 0.761 | +0.023 | 49.9% | 0.495 ±0.012 |
| Federated + DP-SGD | 1 | 0.92 | 0.672 ±0.005 | 0.672 | -0.000 | 50.3% | 0.494 ±0.004 |

Attack accuracy 50% = chance on a balanced pool. Every DP run spends less than the
budget it promised.

### Interpretation

Membership leakage is the generalisation gap. The pooled model scores 100% on its own
training records and 72.8% on unseen ones. That 27-point gap is what the attacker detects,
and it tracks attack success across every configuration:

```
gap +0.272  ->  attacker 70.2%
gap +0.133  ->  attacker 55.5%
gap +0.005  ->  attacker 50.3%   (chance)
```

At ε = 8 the protected model is more accurate than the unprotected federated one (0.785
vs 0.738) while reducing the attacker to chance. Gradient clipping and Gaussian noise
regularise a model that would otherwise memorise. The cost appears at ε = 1: 11.3 accuracy
points for an eight-fold stronger guarantee.

Federated learning halves the leak (0.672 -> 0.565 AUC) but the attacker still beats
chance by 5.5 points. Only DP moves it to 50%.

## How it works

**Data.** 4,250 synthetic citizens across three authorities plus a federal pool, with
Emirati names, IDs and regions. Feature marginals are fitted to the public UCI Heart
Disease cohort (297 anonymous records). Labels are drawn from a logistic model trained on
that real data. Each entity gets a deliberately different population (Abu Dhabi older and
more male-skewed, Dubai younger, Sharjah in between).

| entity | trains on | evaluates | held out | mean age | risk rate |
|---|---|---|---|---|---|
| Abu Dhabi Health | 765 | 135 | 600 | 57.1 | 54.4% |
| Dubai Health Authority | 765 | 135 | 600 | 50.7 | 43.3% |
| Sharjah Health | 637 | 113 | 500 | 54.0 | 48.7% |

**Federation.** 12 rounds x 6 local epochs. Each client owns its own model, optimiser,
data loader and privacy engine; only `state_dict` tensors reach the aggregator, which
averages them weighted by sample count. No raw record crosses a boundary.

**Privacy.** Opacus DP-SGD: per-sample gradient clipping at C = 1, Gaussian noise, Poisson
subsampling. σ is binary-searched so the Rényi accountant reports ε ≤ target over the full
432-step budget. The achieved ε is re-read from the engine after training.

ε and σ are computed per entity and differ because the entities differ in size:

| entity | records | σ at ε = 8 | ε spent |
|---|---|---|---|
| Abu Dhabi | 765 | 2.37 | 7.40 |
| Dubai | 765 | 2.37 | 7.40 |
| Sharjah | 637 | 2.57 | 7.41 |

Sharjah is smaller, so each batch is a larger fraction of its data and it must add more
noise for the same guarantee. The worst case across entities is reported as the system's ε.

**Attack.** A black-box score-based membership-inference attack. For each candidate it
extracts four features from the model's output (confidence in the true label, max
confidence, margin, entropy) and trains a logistic classifier to separate members from
non-members, scored by 5-fold out-of-fold cross-validation so every candidate is
predicted by a model that never saw it.

## Methodology

**1. Non-members are distribution-matched.** Each entity generates a `holdout` split that
never enters training under any configuration. Attack non-members are drawn only from
those holdouts, in equal numbers per entity to the members. Without this control an MIA
evaluation silently measures distribution shift rather than membership leakage.

The shifted comparison is retained as a labelled ablation (`attack_auc_shifted`), using
the federal pool (which has different age, cholesterol and prevalence parameters) as
non-members. The two agree within ±0.015 AUC:

| configuration | matched non-members | shifted non-members | difference |
|---|---|---|---|
| Centralised | 0.672 | 0.661 | -0.010 |
| Federated | 0.565 | 0.550 | -0.015 |
| DP, ε = 8 | 0.505 | 0.497 | -0.008 |
| DP, ε = 1 | 0.494 | 0.497 | +0.003 |

The conclusions do not depend on the choice.

**2. The pool is balanced.** 1,500 members vs 1,500 non-members, so chance is exactly 50%.
On an imbalanced pool accuracy would track the majority class.

**3. TPR at low FPR is reported.** Average-case AUC understates worst-case risk: a metric
near 0.5 can hide individuals identified with near-certainty. The modern standard since
Carlini et al. (2022) is TPR at a low false-positive rate. Ours is 1.4% at 1% FPR for the
centralised baseline (barely above the 1% chance line). This attack recovers membership on
average but does not identify any single individual with high confidence. A stronger
per-example-calibrated attack (LiRA) would likely raise that figure for the unprotected
baselines. The DP guarantee bounds any attacker, not just the one implemented here.

**4. Error bars reflect attack variance.** The candidate pool and cross-validation folds
are re-seeded per repeat (`ATTACK_SEED_BASE + rep`), so the ± figures capture attack
variance, not just model variance. The seed depends on the repeat only (never on the
architecture), so all six configurations face an identical pool and are directly comparable.

**5. The example citizen is selected by rule.** It is the member the baseline attack
scores highest, chosen automatically and reproducibly.

## Applicability to other government data

The domain lives entirely in `data.py`, in four declarations:

| declaration | meaning |
|---|---|
| `CONT`, `BINARY`, `MULTI` | continuous, binary and categorical feature names |
| `CAT_LEVELS` | the permitted values of each categorical |
| `TARGET` | the binary outcome to predict |
| `ENTITY_PARAMS` | one entry per organisation: population size, held-out size, and population differences |

Everything downstream (the design matrix, the model, FedAvg, DP accounting, the attack)
reads those declarations and is domain-blind. The same approach applies to:

- **Financial** (banks jointly training fraud or credit-risk models without pooling
  transactions)
- **Tax and benefits** (cross-department eligibility or fraud analytics where records
  are legally siloed)
- **Immigration and labour** (risk scoring across authorities that cannot share case files)
- **Education and policing** (outcome prediction across emirate-level entities)

The privacy argument transfers unchanged: ε is a property of the training procedure, not
the subject matter.

## Known limitations

- **Synthetic features use independent marginals.** Real clinical variables are correlated;
  ours are not. Achievable accuracy is capped around 0.78. The comparison across
  configurations is unaffected since every configuration sees the same data.
- **One attack.** Attack results are a lower bound on leakage. The DP guarantee is a bound
  on all attackers, not a measurement of one.
- **`secure_mode` is off**, using non-secure RNG for reproducibility. Production
  deployments must retrain with it enabled.
- **ε is per entity, per model.** Training many models on the same people requires
  composing budgets across all of them.
- **No threat model for the aggregator's inputs.** Protection against a malicious
  aggregator inspecting individual updates would require secure aggregation, which is
  complementary and out of scope.
- Results are specific to one tabular regime and one model size.

## Project layout

| path | role |
|---|---|
| `privacy_mesh/data.py` | synthetic populations, entity partitions, train/test/holdout splits |
| `privacy_mesh/model.py` | the shared classifier |
| `privacy_mesh/privacy.py` | ε -> σ calibration against the Rényi accountant |
| `privacy_mesh/federation.py` | FedAvg with per-entity DP-SGD |
| `privacy_mesh/attack.py` | membership-inference attack |
| `privacy_mesh/experiment.py` | orchestration, seeding, result cache |
| `privacy_mesh/charts.py`, `demo.py`, `app.py` | figures and Gradio dashboard |
| `tests/` | property tests including the privacy-budget check |
| `results/` | precomputed runs |
| `data/` | vendored UCI heart-disease cohort |

## Commands

```bash
python -m privacy_mesh.cli prewarm --force       # rebuild the grid and all charts
python -m privacy_mesh.cli serve                 # Gradio dashboard on :7860
python -m pytest tests/ -q                       # property tests
```

Data attribution: UCI Heart Disease (Cleveland), Detrano et al., donated 1988. Used only
to calibrate synthetic statistics. No real personal data appears in this project.