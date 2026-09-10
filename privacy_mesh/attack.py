import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from privacy_mesh.data import ENTITY_KEYS, to_matrix, to_target
from privacy_mesh.model import logits

ID_COLUMNS = ["citizen_id", "citizen_name", "entity", "region"]
FPR_TARGETS = [0.01, 0.001]


def _scores(model, x_np, y_np):
    z = logits(model, x_np)
    p = 1.0 / (1.0 + np.exp(-z))
    conf_true = np.where(y_np == 1, p, 1.0 - p)
    conf_max = np.maximum(p, 1.0 - p)
    margin = np.abs(2.0 * p - 1.0)
    eps = 1e-12
    entropy = -(p * np.log(p + eps) + (1.0 - p) * np.log(1.0 - p + eps))
    return np.stack([conf_true, conf_max, margin, entropy], axis=1)


def _take(df, n, rng):
    take = min(n, len(df))
    idx = rng.choice(len(df), size=take, replace=False)
    return df.iloc[idx]


def build_candidates(parts, n_members=900, n_nonmembers=900, seed=0):
    ents = parts["entities"]
    rng = np.random.default_rng(seed)
    per_member = max(1, n_members // len(ENTITY_KEYS))
    per_non = max(1, n_nonmembers // len(ENTITY_KEYS))
    members = []
    nonmembers = []
    for key in ENTITY_KEYS:
        members.append(_take(ents[key]["train"], per_member, rng))
        nonmembers.append(_take(ents[key]["holdout"], per_non, rng))
    return (
        pd.concat(members, ignore_index=True),
        pd.concat(nonmembers, ignore_index=True),
    )


def build_shifted_candidates(parts, n_members=900, n_nonmembers=900, seed=0):
    ents = parts["entities"]
    rng = np.random.default_rng(seed)
    per_member = max(1, n_members // len(ENTITY_KEYS))
    members = [_take(ents[key]["train"], per_member, rng) for key in ENTITY_KEYS]
    nonmembers = [_take(ents["external"], n_nonmembers, rng)]
    return (
        pd.concat(members, ignore_index=True),
        pd.concat(nonmembers, ignore_index=True),
    )


def _tpr_at_fpr(fpr, tpr, target):
    mask = fpr <= target
    if not mask.any():
        return 0.0
    return float(tpr[mask].max())


def _roc(prob, true):
    order = np.argsort(-prob)
    sorted_true = true[order]
    n_pos = int(sorted_true.sum())
    n_neg = len(sorted_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    tps = np.cumsum(sorted_true)
    fps = np.cumsum(1 - sorted_true)
    tpr = np.concatenate([[0.0], tps / n_pos, [1.0]])
    fpr = np.concatenate([[0.0], fps / n_neg, [1.0]])
    return fpr, tpr


def _run_attack_model(sm, label, seed, frame=None, folds=5):
    prob = np.zeros(len(sm), dtype=np.float64)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed % (2 ** 31))
    for fit_idx, score_idx in splitter.split(sm, label):
        scaler = StandardScaler().fit(sm[fit_idx])
        clf = LogisticRegression(max_iter=2000)
        clf.fit(scaler.transform(sm[fit_idx]), label[fit_idx])
        prob[score_idx] = clf.predict_proba(scaler.transform(sm[score_idx]))[:, 1]
    fpr, tpr = _roc(prob, label)
    out = {
        "attack_auc": float(roc_auc_score(label, prob)),
        "attack_acc": float(accuracy_score(label, (prob >= 0.5).astype(int))),
        "roc_fpr": fpr,
        "roc_tpr": tpr,
        "conf_members": sm[label == 1][:, 0],
        "conf_nonmembers": sm[label == 0][:, 0],
        "n_candidates": int(len(sm)),
        "n_eval": int(len(sm)),
        "folds": folds,
    }
    for t in FPR_TARGETS:
        out[f"tpr_at_fpr_{t:g}"] = _tpr_at_fpr(fpr, tpr, t)
    if frame is not None:
        verdicts = frame[ID_COLUMNS].copy()
        verdicts["truth"] = label.astype(int)
        verdicts["attack_prob"] = prob
        verdicts["verdict"] = (prob >= 0.5).astype(int)
        out["verdicts"] = verdicts
    return out


def run_attack(model, parts, mean, std, seed=0, n_members=900, n_nonmembers=900,
               include_shifted=True):
    members, nonmembers = build_candidates(parts, n_members, n_nonmembers, seed)
    frame = pd.concat([members, nonmembers], ignore_index=True)
    label = np.concatenate([np.ones(len(members)), np.zeros(len(nonmembers))])
    sm = _scores(model, to_matrix(frame, mean, std), to_target(frame))
    result = {"matched": _run_attack_model(sm, label, seed, frame)}
    if include_shifted:
        sm_members, sm_non = build_shifted_candidates(parts, n_members, n_nonmembers, seed)
        sframe = pd.concat([sm_members, sm_non], ignore_index=True)
        slabel = np.concatenate([np.ones(len(sm_members)), np.zeros(len(sm_non))])
        ssm = _scores(model, to_matrix(sframe, mean, std), to_target(sframe))
        result["shifted"] = _run_attack_model(ssm, slabel, seed)
    return result


def pick_exemplar(verdicts):
    members = verdicts[verdicts["truth"] == 1]
    if members.empty:
        return None
    return members.sort_values("attack_prob", ascending=False).iloc[0]["citizen_id"]


def lookup_verdict(verdicts, citizen_id):
    hit = verdicts[verdicts["citizen_id"] == citizen_id]
    if hit.empty:
        return None
    row = hit.iloc[0]
    return {
        "citizen_id": row["citizen_id"],
        "citizen_name": row["citizen_name"],
        "entity": row["entity"],
        "region": row["region"],
        "truth": int(row["truth"]),
        "attack_prob": float(row["attack_prob"]),
        "verdict": int(row["verdict"]),
        "correct": int(row["verdict"]) == int(row["truth"]),
    }
