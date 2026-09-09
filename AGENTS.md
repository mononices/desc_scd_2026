UAE Privacy Mesh

The idea

UAE Privacy Mesh allows government/health entities to collaboratively train a model without sharing raw citizen data.

The innovative part is simple: we attack our own model to measure whether privacy actually works.

Instead of:

Entity A + Entity B → Central Database

we use:

Entity A → Local Training → Protected Updates → Aggregator ← Protected Updates ← Entity B

---
What the cybersecurity case competition wants:

Privacy-Preserving Analytics on Sensitive Government Data

Difficulty: Advanced
Why this matters

- UAE data residency and PDPL obligations often block the obvious thing: pooling sensitive records to train a model or compute a statistic. Privacy-preserving techniques - differential privacy and federated learning - let organisations get the insight without moving or exposing the underlying personal data.

- What to build

- A demonstrator that computes useful analytics or trains a shared model across sensitive (simulated) datasets while provably limiting what any party - including the aggregator - can learn about an individual.

Minimum deliverable (what the competition grades)

- A realistic synthetic dataset representing personal records held across two or more 'entities' (e.g. hospitals, departments) that are not allowed to share raw data.
One core technique implemented correctly: either differential privacy (a mechanism with a stated, correctly-applied epsilon budget) or federated learning (local training, only updates shared), or both combined.
A concrete task with a measurable answer: a population statistic, a query interface, or a simple classifier trained across the entities.
A privacy demonstration: a membership-inference or reconstruction attempt against the output that succeeds on the naive baseline and fails (or degrades) under your technique.
The utility/privacy trade-off shown explicitly - accuracy or error as a function of the privacy budget, with the assumptions and limits stated.

What the demo should show

- Run the naive version and show an attacker recovering whether a specific individual was in the data; switch on the privacy technique, show the attack fail, and show the analytic answer is still useful with the accuracy cost quantified.

---

What we build

1. Synthetic Government Data

Create realistic synthetic health records for 2–3 simulated health authorities, with slightly different populations.

Example features:

- Age
- Region
- Health-condition flags
- Risk factors
- Binary health-risk outcome

2. Federated Learning

Each entity trains the same small PyTorch model locally.

Only model updates are shared with the aggregator using Federated Averaging (FedAvg).

3. Differential Privacy

Add DP-SGD using Opacus before updates leave each entity.

We test different privacy budgets:

ε = ∞, 8, 4, 2, 1

This demonstrates the real privacy/accuracy trade-off.

4. Privacy Red Team

Instead of simply claiming that the system is private, we try to attack it.

A simple membership-inference attack attempts to determine:

«“Was this individual included in the training data?”»

We compare:

Centralized → FL → FL + DP

The expected result is that the attack becomes less effective as privacy protection increases.

---

Interactive Demo

Build the final demo using Gradio.

The jury can select:

Architecture

- Centralized
- Federated
- Federated + DP

Privacy budget

- ∞
- 8
- 4
- 2
- 1

Then click Run Experiment.

The dashboard shows:

- Model accuracy
- Membership-inference attack AUC
- Privacy budget ε
- Privacy/utility chart

The key message:

«“We sacrifice a small amount of model utility to make individual membership increasingly difficult to detect.”»

---

Technical Stack

- Python 3.11+
- PyTorch — ML model
- Flower — Federated Learning
- Opacus — Differential Privacy
- NumPy + pandas + Faker — synthetic data
- scikit-learn — attack + evaluation metrics
- Matplotlib — charts
- Gradio — interactive demo

Use this dataset for demonstration: https://archive.ics.uci.edu/dataset/45/heart+disease
Use this model for training: https://huggingface.co/nexusbert/heart-disease-random-forest (RandomForestClassifier)

No custom cryptography, complicated models, or unnecessary infrastructure, dont leave any comments in the code

---

3-Day Build Plan

Day 1 — Core system

- Generate synthetic datasets
- Build PyTorch classifier
- Implement Flower/FedAvg
- Get centralized + federated baselines working

Day 2 — Privacy

- Add Opacus DP-SGD
- Implement membership-inference attack
- Run ε sweep
- Generate results

Day 3 — Demo + submission

- Build Gradio interface
- Add charts and comparison
- Test clean end-to-end execution
- Prepare the ≤5-page PDF

---

One-line pitch

«UAE Privacy Mesh enables government entities to learn together without sharing citizen data — and then attacks the model to prove how much privacy it actually provides.»can 