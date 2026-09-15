import hashlib
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
PART_DIR = os.path.join(DATA_DIR, "partitions")
REAL_CSV = os.path.join(DATA_DIR, "heart_cleveland.csv")

CONT = ["age", "trestbps", "chol", "thalach", "oldpeak"]
BINARY = ["sex", "fbs", "exang"]
MULTI = ["cp", "restecg", "slope", "ca", "thal"]
FEATURES = CONT + BINARY + MULTI
TARGET = "num"

GIVEN_NAMES_M = [
    "Ahmed", "Mohammed", "Khalid", "Saeed", "Rashid", "Hamdan", "Sultan", "Omar",
    "Abdullah", "Faisal", "Tariq", "Yousef", "Majid", "Nasser", "Salem", "Hassan",
    "Ali", "Mansour", "Zayed", "Juma", "Obaid", "Marwan", "Waleed", "Adel",
]
GIVEN_NAMES_F = [
    "Fatima", "Aisha", "Mariam", "Noura", "Hessa", "Latifa", "Shamma", "Alia",
    "Maitha", "Sara", "Amna", "Reem", "Salama", "Moza", "Wadeema", "Hind",
    "Khawla", "Budour", "Dana", "Shaikha", "Asma", "Rawda", "Muna", "Yasmin",
]
FAMILY_NAMES = [
    "Al Mansoori", "Al Nuaimi", "Al Marzooqi", "Al Shamsi", "Al Zaabi", "Al Hammadi",
    "Al Suwaidi", "Al Ketbi", "Al Dhaheri", "Al Blooshi", "Al Falasi", "Al Qubaisi",
    "Al Rumaithi", "Al Ameri", "Al Kaabi", "Al Mazrouei", "Al Hosani", "Al Khoori",
    "Al Naqbi", "Al Sharqi", "Al Awadhi", "Al Raisi", "Al Balushi", "Al Jaberi",
]

CAT_LEVELS = {
    "cp": [1, 2, 3, 4],
    "restecg": [0, 1, 2],
    "slope": [1, 2, 3],
    "ca": [0, 1, 2, 3],
    "thal": [3, 6, 7],
}

ENTITY_KEYS = ["abu_dhabi", "dubai", "sharjah"]
ENTITY_LABELS = {
    "abu_dhabi": "Abu Dhabi Health",
    "dubai": "Dubai Health Authority",
    "sharjah": "Sharjah Health",
}
REGIONS = {
    "abu_dhabi": [("Abu Dhabi City", 0.55), ("Al Ain", 0.30), ("Al Dhafra", 0.15)],
    "dubai": [("Dubai City", 0.60), ("Deira", 0.25), ("Jebel Ali", 0.15)],
    "sharjah": [("Sharjah City", 0.62), ("Khor Fakkan", 0.23), ("Kalba", 0.15)],
}
ENTITY_PARAMS = {
    "abu_dhabi": {
        "size": 900, "holdout": 600, "age_shift": 2.5, "chol_shift": 4.0,
        "thalach_shift": -3.0, "male_p": 0.70, "intercept_shift": 0.30, "seed": 1001,
    },
    "dubai": {
        "size": 900, "holdout": 600, "age_shift": -3.5, "chol_shift": 9.0,
        "thalach_shift": 4.0, "male_p": 0.62, "intercept_shift": -0.30, "seed": 1002,
    },
    "sharjah": {
        "size": 750, "holdout": 500, "age_shift": 0.0, "chol_shift": -7.0,
        "thalach_shift": 0.0, "male_p": 0.66, "intercept_shift": -0.05, "seed": 1003,
    },
}
EXTERNAL_PARAMS = {
    "size": 700, "age_shift": 0.0, "chol_shift": 0.0, "thalach_shift": 0.0,
    "male_p": 0.68, "intercept_shift": 0.0, "seed": 1099,
    "regions": [("Federal MOH Pool", 1.0)],
}
TEST_FRAC = 0.15
GLOBAL_SEED = 0
DATA_VERSION = 2


def _stats(df):
    stats = {}
    for c in CONT:
        v = df[c].to_numpy(dtype=float)
        stats[c] = {"mean": float(v.mean()), "std": float(v.std())}
    zero = df["oldpeak"].to_numpy(dtype=float)
    stats["oldpeak_zero_p"] = float((zero < 0.05).mean())
    nz = zero[zero >= 0.05]
    stats["oldpeak_nz_mean"] = float(nz.mean())
    stats["oldpeak_nz_std"] = float(nz.std())
    for c in BINARY + MULTI:
        vc = df[c].value_counts(normalize=True).sort_index()
        stats[c] = {str(int(k)): float(p) for k, p in vc.items()}
    return stats


def _sample_stats(stats, n, rng):
    out = {}
    out["age"] = np.clip(rng.normal(stats["age"]["mean"], stats["age"]["std"], n), 29, 77)
    out["trestbps"] = np.clip(rng.normal(stats["trestbps"]["mean"], stats["trestbps"]["std"], n), 94, 200)
    out["chol"] = np.clip(rng.normal(stats["chol"]["mean"], stats["chol"]["std"], n), 100, 450)
    out["thalach"] = np.clip(rng.normal(stats["thalach"]["mean"], stats["thalach"]["std"], n), 71, 210)
    op = np.zeros(n)
    mask = rng.random(n) > stats["oldpeak_zero_p"]
    op[mask] = np.clip(
        rng.normal(stats["oldpeak_nz_mean"], stats["oldpeak_nz_std"], n)[mask], 0.05, 6.2
    )
    out["oldpeak"] = np.round(op, 1)
    for c in BINARY + MULTI:
        cats = np.array([int(k) for k in stats[c]])
        probs = np.array(list(stats[c].values()))
        out[c] = rng.choice(cats, size=n, p=probs)
    return out


def _cat_columns():
    return [f"{c}_{k}" for c in MULTI for k in CAT_LEVELS[c]]


CAT_COLUMNS = _cat_columns()
MODEL_COLUMNS = CONT + BINARY + CAT_COLUMNS


def _design_matrix(df):
    parts = []
    for c in CONT + BINARY:
        parts.append(df[c].to_numpy(dtype=np.float64))
    for c in MULTI:
        col = df[c].to_numpy(dtype=float)
        for k in CAT_LEVELS[c]:
            parts.append((col == k).astype(np.float64))
    return np.stack(parts, axis=1)


def _config_hash():
    cfg = {"entities": ENTITY_PARAMS, "external": EXTERNAL_PARAMS, "test_frac": TEST_FRAC,
           "seed": GLOBAL_SEED, "version": DATA_VERSION}
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]


def _normed_lr(df):
    x = _design_matrix(df)
    sc = StandardScaler().fit(x)
    y = df[TARGET].to_numpy(dtype=np.float64)
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(x), y)
    return sc, lr


def _names(sex, rng):
    given_m = rng.choice(GIVEN_NAMES_M, size=len(sex))
    given_f = rng.choice(GIVEN_NAMES_F, size=len(sex))
    family = rng.choice(FAMILY_NAMES, size=len(sex))
    given = np.where(sex == 1, given_m, given_f)
    return [f"{g} {f}" for g, f in zip(given, family)]


def _sample_entity(key, params, stats, lr, sc, entity_index):
    n = params["size"] + params.get("holdout", 0)
    rng = np.random.default_rng(params["seed"] + GLOBAL_SEED)
    data = _sample_stats(stats, n, rng)
    data["age"] = np.clip(data["age"] + params["age_shift"], 29, 80)
    data["chol"] = np.clip(data["chol"] + params["chol_shift"], 100, 450)
    data["thalach"] = np.clip(data["thalach"] + params["thalach_shift"], 71, 210)
    male_p = params["male_p"]
    if male_p != stats["sex"]["1"]:
        idx = rng.random(n) < abs(male_p - stats["sex"]["1"])
        data["sex"][idx] = 1 if male_p > stats["sex"]["1"] else 0
    df = pd.DataFrame(data)
    for c in CONT + BINARY + MULTI:
        if c in ("age", "trestbps", "chol", "thalach"):
            df[c] = df[c].round().astype(int)
    x = _design_matrix(df)
    z = lr.intercept_[0] + lr.coef_[0] @ sc.transform(x).T + params["intercept_shift"]
    p = 1.0 / (1.0 + np.exp(-z))
    df[TARGET] = (rng.random(n) < p).astype(int)
    regions = REGIONS[key] if key in REGIONS else params["regions"]
    names = [r for r, _ in regions]
    weights = [w for _, w in regions]
    df["region"] = rng.choice(names, size=n, p=weights)
    prefix = "EXT" if key == "external" else key[:2].upper()
    df["citizen_id"] = [f"{prefix}-{entity_index:02d}-{i:06d}" for i in range(n)]
    df["citizen_name"] = _names(df["sex"].to_numpy(), rng)
    df["entity"] = key
    cols = ["citizen_id", "citizen_name", "entity", "region"] + FEATURES + [TARGET]
    return df[cols]


def ensure_partitions(force=False):
    os.makedirs(PART_DIR, exist_ok=True)
    meta_path = os.path.join(PART_DIR, "meta.json")
    cur_hash = _config_hash()
    if not force and os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("hash") == cur_hash:
                return _load_partitions(meta)
        except (json.JSONDecodeError, OSError, KeyError, ValueError):
            pass
    if not os.path.exists(REAL_CSV):
        raise FileNotFoundError(
            f"reference cohort missing: {REAL_CSV}\n"
            "The UCI heart-disease CSV ships with this repository; restore it from the "
            "submission bundle or re-clone before running."
        )
    real = pd.read_csv(REAL_CSV)
    stats = _stats(real)
    sc, lr = _normed_lr(real)
    partitions = {}
    ei = 0
    for key in ENTITY_KEYS:
        params = ENTITY_PARAMS[key]
        df = _sample_entity(key, params, stats, lr, sc, ei)
        ei += 1
        seed = GLOBAL_SEED + params["seed"]
        enrolled, holdout = train_test_split(
            df, test_size=params["holdout"], random_state=seed, stratify=df[TARGET],
        )
        train, test = train_test_split(
            enrolled, test_size=TEST_FRAC, random_state=seed, stratify=enrolled[TARGET],
        )
        partitions[key] = {
            "train": train.reset_index(drop=True),
            "test": test.reset_index(drop=True),
            "holdout": holdout.reset_index(drop=True),
        }
        for split, part in partitions[key].items():
            part.to_csv(os.path.join(PART_DIR, f"{key}_{split}.csv"), index=False)
    ext = _sample_entity("external", EXTERNAL_PARAMS, stats, lr, sc, 99)
    partitions["external"] = ext.reset_index(drop=True)
    ext.to_csv(os.path.join(PART_DIR, "external.csv"), index=False)
    meta = {
        "hash": cur_hash,
        "mean": sc.mean_.tolist(),
        "std": sc.scale_.tolist(),
        "real_mean": {c: stats[c]["mean"] for c in CONT},
        "stats": {c: stats[c] for c in BINARY + MULTI},
        "prevalence": {k: float(p["train"][TARGET].mean()) for k, p in partitions.items() if k != "external"},
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return _load_partitions(meta)


def _load_partitions(meta):
    parts = {}
    for key in ENTITY_KEYS:
        parts[key] = {
            split: pd.read_csv(os.path.join(PART_DIR, f"{key}_{split}.csv"))
            for split in ("train", "test", "holdout")
        }
    parts["external"] = pd.read_csv(os.path.join(PART_DIR, "external.csv"))
    real = pd.read_csv(REAL_CSV)
    parts["real"] = real
    return {
        "entities": parts,
        "mean": np.asarray(meta["mean"], dtype=np.float64),
        "std": np.asarray(meta["std"], dtype=np.float64),
        "meta": meta,
    }


def to_matrix(df, mean, std):
    x = _design_matrix(df)
    return ((x - mean) / std).astype(np.float32)


def to_target(df):
    return df[TARGET].to_numpy(dtype=np.int64)
