import hashlib
import json
import os
import time

import numpy as np
import torch

from privacy_mesh import attack as atk
from privacy_mesh import data
from privacy_mesh import federation, model as mdl

ROOT = data.ROOT
RESULTS_DIR = os.path.join(ROOT, "results")
MODEL_DIR = os.path.join(RESULTS_DIR, "models")
CHART_DIR = os.path.join(RESULTS_DIR, "charts")

DELTA = 1e-5
ROUNDS = 12
LOCAL_EPOCHS = 6
BATCH = 128
LR = 0.05
CENTRAL_EPOCHS = 150
CLIP = 1.0
N_MEMBERS = 800
N_NONMEMBERS = 1100
EPSILON_OPTIONS = [8.0, 4.0, 2.0, 1.0]
PREWARM_REPS = 3
REP_SEED_GAP = 7919

ARCH_LABELS = {
    "centralized": "Centralised (data pooled)",
    "federated": "Federated (FedAvg)",
    "fed_dp": "Federated + DP-SGD",
}


def result_key(arch, eps=None):
    tag = "none" if eps is None else f"{eps:g}"
    return f"{arch}__{tag}"


def record_key(arch, eps, rep):
    return f"{result_key(arch, eps)}__r{rep}"


def run_seed(arch, eps, rep=0):
    raw = f"{arch}|{eps}|{rep}"
    return int(hashlib.sha1(raw.encode()).hexdigest(), 16) % (2 ** 31)


def _entity_matrices(parts, mean, std):
    ents = parts["entities"]
    out = []
    for key in data.ENTITY_KEYS:
        tr = ents[key]["train"]
        out.append((data.to_matrix(tr, mean, std), data.to_target(tr)))
    test = [ents[k]["test"] for k in data.ENTITY_KEYS]
    pool = test[0]._append(test[1:], ignore_index=True)
    return out, data.to_matrix(pool, mean, std), data.to_target(pool)


def run_experiment(arch, eps=None, quick=False, progress=None, rep=0):
    cfg = dict(rounds=ROUNDS, local_epochs=LOCAL_EPOCHS, batch=BATCH, lr=LR,
               central_epochs=CENTRAL_EPOCHS, n_members=N_MEMBERS,
               n_nonmembers=N_NONMEMBERS)
    if quick:
        cfg.update(rounds=4, central_epochs=40, n_members=300, n_nonmembers=400)
    if eps is not None and arch != "fed_dp":
        eps = None
    parts = data.ensure_partitions()
    entity_train, x_test, y_test = _entity_matrices(parts, parts["mean"], parts["std"])
    real_df = parts["entities"]["real"]
    x_real = data.to_matrix(real_df, parts["mean"], parts["std"])
    y_real = data.to_target(real_df)
    seed = run_seed(arch, eps, rep)
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    t0 = time.time()
    eps_achieved = None
    sigma = None
    if arch == "centralized":
        x_all = np.vstack([x for x, _ in entity_train])
        y_all = np.concatenate([y for _, y in entity_train])

        def cb(frac, msg):
            if progress:
                progress(0.05 + 0.75 * frac, msg)

        model, history = mdl.train_central(
            x_all, y_all, x_test, y_test,
            epochs=cfg["central_epochs"], batch=cfg["batch"], lr=cfg["lr"],
            seed=seed, progress=cb,
        )
        n_train = len(x_all)
        rounds = cfg["central_epochs"]
    else:
        def cb(frac, msg):
            if progress:
                progress(0.05 + 0.75 * frac, msg)

        res = federation.train_federated(
            entity_train, x_test, y_test,
            rounds=cfg["rounds"], local_epochs=cfg["local_epochs"],
            batch=cfg["batch"], lr=cfg["lr"], eps=eps, delta=DELTA,
            clip=CLIP, seed=seed, progress=cb,
        )
        model = res["model"]
        history = res["history"]
        eps_achieved = res["eps_achieved"]
        sigma = res["sigma"]
        n_train = sum(len(x) for x, _ in entity_train)
        rounds = cfg["rounds"]
    acc, auc = mdl.evaluate(model, x_test, y_test)
    real_acc, real_auc = mdl.evaluate(model, x_real, y_real)
    if progress:
        progress(0.82, "launching membership-inference attack")
    attack = atk.run_attack(
        model, parts, parts["mean"], parts["std"],
        seed=0, n_members=cfg["n_members"], n_nonmembers=cfg["n_nonmembers"],
    )["full"]
    elapsed = time.time() - t0
    os.makedirs(MODEL_DIR, exist_ok=True)
    base_key = result_key(arch, eps)
    key = record_key(arch, eps, rep)
    model_file = f"{key}.pt"
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, model_file))
    record = {
        "key": key,
        "config": base_key,
        "rep": rep,
        "arch": arch,
        "arch_label": ARCH_LABELS[arch],
        "eps_target": eps,
        "eps_achieved": eps_achieved,
        "sigma": sigma,
        "delta": DELTA,
        "acc": float(acc),
        "auc": float(auc),
        "real_acc": float(real_acc),
        "real_auc": float(real_auc),
        "attack_auc": float(attack["attack_auc"]),
        "attack_acc": float(attack["attack_acc"]),
        "n_candidates": int(attack["n_candidates"]),
        "roc_fpr": attack["roc_fpr"].tolist(),
        "roc_tpr": attack["roc_tpr"].tolist(),
        "conf_members": attack["conf_members"].tolist(),
        "conf_nonmembers": attack["conf_nonmembers"].tolist(),
        "n_train": n_train,
        "rounds": rounds,
        "history": history,
        "time_s": round(elapsed, 1),
        "seed": seed,
        "model_file": model_file,
        "quick": quick,
    }
    save_record(record)
    if progress:
        progress(1.0, "done")
    return record


def results_path():
    return os.path.join(RESULTS_DIR, "results.json")


def load_records():
    path = results_path()
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        items = json.load(f)
    return {r["key"]: r for r in items}


def save_record(record):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    records = load_records()
    records[record["key"]] = record
    with open(results_path(), "w") as f:
        json.dump(list(records.values()), f, indent=1)


def group_records(records=None):
    if records is None:
        records = load_records()
    groups = {}
    for rec in records.values():
        groups.setdefault(rec["config"], []).append(rec)
    for g in groups.values():
        g.sort(key=lambda r: r["rep"])
    return groups


def summarize(records):
    base = dict(records[0])
    n = len(records)
    base["n_reps"] = n
    base["rep"] = -1
    base["key"] = base["config"]
    for field in ("acc", "auc", "real_acc", "real_auc", "attack_auc", "attack_acc", "time_s"):
        vals = [r[field] for r in records]
        base[f"{field}_std"] = float(np.std(vals)) if n > 1 else 0.0
        base[field] = float(np.mean(vals))
    eps_vals = [r["eps_achieved"] for r in records if r["eps_achieved"] is not None]
    if eps_vals:
        base["eps_achieved"] = float(np.mean(eps_vals))
        base["eps_achieved_std"] = float(np.std(eps_vals)) if len(eps_vals) > 1 else 0.0
    sig_vals = [r["sigma"] for r in records if r["sigma"] is not None]
    if sig_vals:
        base["sigma"] = float(np.mean(sig_vals))
    base["roc_fpr"] = records[0]["roc_fpr"]
    base["roc_tpr"] = records[0]["roc_tpr"]
    base["conf_members"] = records[0]["conf_members"]
    base["conf_nonmembers"] = records[0]["conf_nonmembers"]
    base["history"] = records[0]["history"]
    return base


def config_summaries():
    return {cfg: summarize(recs) for cfg, recs in group_records().items()}


def sweep_configs():
    cfgs = [("centralized", None), ("federated", None)]
    for e in EPSILON_OPTIONS:
        cfgs.append(("fed_dp", e))
    return cfgs
