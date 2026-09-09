import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from privacy_mesh.data import ENTITY_KEYS, to_matrix, to_target
from privacy_mesh.model import logits


def _scores(model, x_np, y_np):
    z = logits(model, x_np)
    p = 1.0 / (1.0 + np.exp(-z))
    conf_true = np.where(y_np == 1, p, 1.0 - p)
    conf_max = np.maximum(p, 1.0 - p)
    margin = np.abs(2.0 * p - 1.0)
    eps = 1e-12
    entropy = -(p * np.log(p + eps) + (1.0 - p) * np.log(1.0 - p + eps))
    return np.stack([conf_true, conf_max, margin, entropy], axis=1)


def build_candidates(parts, mean, std, n_members=800, n_nonmembers=1100, seed=0):
    ents = parts["entities"]
    rng = np.random.default_rng(seed)
    member_rows = []
    nonmember_rows = []
    member_src = []
    nonmember_src = []
    per_entity_members = max(1, n_members // len(ENTITY_KEYS))
    per_entity_non = max(1, n_nonmembers // (len(ENTITY_KEYS) + 1))
    for key in ENTITY_KEYS:
        df = ents[key]["train"]
        take = min(per_entity_members, len(df))
        idx = rng.choice(len(df), size=take, replace=False)
        member_rows.append(df.iloc[idx])
        member_src.append(np.full(take, key))
        te = ents[key]["test"]
        take_te = min(per_entity_non, len(te))
        idx = rng.choice(len(te), size=take_te, replace=False)
        nonmember_rows.append(te.iloc[idx])
        nonmember_src.append(np.full(take_te, key))
    ext = ents["external"]
    taken = sum(len(nr) for nr in nonmember_rows)
    take_ext = min(n_nonmembers - taken, len(ext))
    idx = rng.choice(len(ext), size=take_ext, replace=False)
    nonmember_rows.append(ext.iloc[idx])
    members = member_rows[0]._append(member_rows[1:], ignore_index=True)
    nonmembers = nonmember_rows[0]._append(nonmember_rows[1:], ignore_index=True)
    return members, nonmembers, np.concatenate(member_src)


def _run_attack_model(sm, label, seed):
    rng = np.random.default_rng(seed + 7)
    order = rng.permutation(len(sm))
    sm = sm[order]
    label = label[order]
    split = int(len(sm) * 0.6)
    scaler = StandardScaler().fit(sm[:split])
    clf = LogisticRegression(max_iter=2000)
    clf.fit(scaler.transform(sm[:split]), label[:split])
    prob = clf.predict_proba(scaler.transform(sm[split:]))[:, 1]
    true = label[split:]
    attack_auc = roc_auc_score(true, prob)
    attack_acc = accuracy_score(true, (prob >= 0.5).astype(int))
    fpr, tpr = _roc(prob, true)
    conf_members = sm[split:][true == 1][:, 0]
    conf_non = sm[split:][true == 0][:, 0]
    return {
        "attack_auc": float(attack_auc),
        "attack_acc": float(attack_acc),
        "roc_fpr": fpr,
        "roc_tpr": tpr,
        "conf_members": conf_members,
        "conf_nonmembers": conf_non,
        "n_candidates": int(len(sm)),
    }


def typicality_scores(x_candidates, x_reference, k=5):
    nn = NearestNeighbors(n_neighbors=min(k, len(x_reference)))
    nn.fit(x_reference)
    dist, _ = nn.kneighbors(x_candidates)
    return dist.mean(axis=1)


def run_attack(model, parts, mean, std, seed=0, n_members=800, n_nonmembers=1100,
               target_frac=0.35):
    members, nonmembers, _ = build_candidates(
        parts, mean, std, n_members=n_members, n_nonmembers=n_nonmembers, seed=seed
    )
    xm = to_matrix(members, mean, std)
    ym = to_target(members)
    xn = to_matrix(nonmembers, mean, std)
    yn = to_target(nonmembers)
    x = np.vstack([xm, xn])
    y = np.concatenate([ym, yn])
    label = np.concatenate([np.ones(len(members)), np.zeros(len(nonmembers))])
    sm = _scores(model, x, y)
    full = _run_attack_model(sm, label, seed)
    x_real_ref = to_matrix(parts["entities"]["real"], mean, std)
    typ = typicality_scores(x, x_real_ref, k=5)
    take = max(40, int(round(target_frac * len(x))))
    targeted_idx = np.argsort(-typ)[:take]
    targeted = _run_attack_model(sm[targeted_idx], label[targeted_idx], seed)
    targeted["typicality_member"] = float(typ[label == 1].mean())
    targeted["typicality_nonmember"] = float(typ[label == 0].mean())
    return {"full": full, "targeted": targeted, "n_candidates": int(len(x)), "n_targeted": take}


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
