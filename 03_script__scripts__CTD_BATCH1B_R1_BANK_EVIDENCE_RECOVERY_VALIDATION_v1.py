# ==============================================================================
# CTD IJDSA REVISION
# BATCH 1B-R1 — BANK MARKETING EVIDENCE RECOVERY VALIDATION v1.0
#
# PURPOSE
# -------
# Rebuild a fully reproducible Bank Marketing evidence block because the
# original Batch 1B train/test index arrays are no longer recoverable.
#
# IMPORTANT
# ---------
# This is VALIDATION ONLY.
# No 300-run production experiment is authorized by this script.
#
# Frozen scientific design:
#   Dataset: Bank Marketing, UCI bank-full.csv
#   Models: logistic_regression, decision_tree, random_forest, linear_svm
#   Seeds: [42, 123]
#   Configurations:
#       baseline
#       scaling_cf
#       sampling_cf
#       representation_cf
#       training_fraction_cf
#   Primary metric: Macro-F1
#   Split: stratified 80/20, seed-specific, persisted exactly
#
# The purpose is to establish:
#   G1 dataset identity
#   G2 exact split persistence
#   G3 preprocessing integrity
#   G4 all model/configuration paths
#   G5 deterministic repeat
#   G6 metric integrity
#   G7 resource feasibility
#   G8 representative agreement with archived Batch 1B values
#
# No scientific claim is changed by this validation.
# ==============================================================================

import gc
import hashlib
import json
import platform
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

# ------------------------------------------------------------------------------
# FROZEN SETTINGS
# ------------------------------------------------------------------------------

BASE = Path("/content/CTD_BATCH1B_R1_VALIDATION")
BASE.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123]
MODELS = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "linear_svm",
]

CONFIGS = {
    "baseline": {
        "scaling": True,
        "sampling": "original",
        "representation": "all_features",
        "training_fraction": 1.0,
    },
    "scaling_cf": {
        "scaling": False,
        "sampling": "original",
        "representation": "all_features",
        "training_fraction": 1.0,
    },
    "sampling_cf": {
        "scaling": True,
        "sampling": "minority_oversampling",
        "representation": "all_features",
        "training_fraction": 1.0,
    },
    "representation_cf": {
        "scaling": True,
        "sampling": "original",
        "representation": "pca95",
        "training_fraction": 1.0,
    },
    "training_fraction_cf": {
        "scaling": True,
        "sampling": "original",
        "representation": "all_features",
        "training_fraction": 0.5,
    },
}

EXPECTED_RUNS = len(SEEDS) * len(MODELS) * len(CONFIGS)

ARCHIVED_PREFIXES = {
    42: "f0a9dd84daff9784",
    123: "59e8cfdd7cba1c6b",
}

# Representative archived Batch 1B values recovered from the execution log.
# These are comparison targets only; they are NOT treated as exact-equivalence
# evidence because the original split arrays are unavailable.
ARCHIVED_REPRESENTATIVES = {
    (42, "logistic_regression", "baseline"):
        {"macro_f1": 0.699278, "balanced_accuracy": 0.661674},
    (42, "logistic_regression", "sampling_cf"):
        {"macro_f1": 0.729043, "balanced_accuracy": 0.832613},
    (42, "logistic_regression", "training_fraction_cf"):
        {"macro_f1": 0.701772, "balanced_accuracy": 0.664322},
    (123, "logistic_regression", "baseline"):
        {"macro_f1": 0.693157, "balanced_accuracy": 0.654295},
}

RESULTS = BASE / "CTD_BATCH1B_R1_VALIDATION_RESULTS.csv"
SPLITS = BASE / "CTD_BATCH1B_R1_VALIDATION_SPLITS.npz"
MANIFEST = BASE / "CTD_BATCH1B_R1_VALIDATION_MANIFEST.json"
REPORT = BASE / "CTD_BATCH1B_R1_VALIDATION_REPORT.json"

print("=" * 78)
print("CTD IJDSA REVISION — BATCH 1B-R1")
print("BANK MARKETING EVIDENCE RECOVERY VALIDATION v1.0")
print("=" * 78)
print("VALIDATION ONLY — NO PRODUCTION AUTHORIZATION")
print()
print("Python       :", platform.python_version())
print("NumPy        :", np.__version__)
print("Pandas       :", pd.__version__)
print("Scikit-learn:", sklearn.__version__)
print("Expected runs:", EXPECTED_RUNS)

# ------------------------------------------------------------------------------
# DATASET
# ------------------------------------------------------------------------------

candidates = [
    Path("/content/bank-full.csv"),
    Path("/content/Bank Marketing/bank-full.csv"),
]

data_path = next((p for p in candidates if p.exists()), None)
if data_path is None:
    raise FileNotFoundError("Validated bank-full.csv not found.")

df = pd.read_csv(data_path, sep=";")

assert df.shape == (45211, 17), df.shape
assert list(df.columns) == [
    "age", "job", "marital", "education", "default", "balance",
    "housing", "loan", "contact", "day", "month", "duration",
    "campaign", "pdays", "previous", "poutcome", "y"
]

assert int(df.isna().sum().sum()) == 0
assert int(df.drop(columns=["y"]).duplicated().sum()) == 0
assert sorted(df["y"].astype(str).str.strip().unique().tolist()) == ["no", "yes"]

X = df.drop(columns=["y"]).copy()
y = df["y"].astype(str).str.strip().map({"no": 0, "yes": 1}).astype(int)

print()
print("DATASET GATE")
print("-" * 78)
print("Path       :", data_path)
print("Shape      :", df.shape)
print("Features   :", X.shape)
print("Target     :", sorted(y.unique().tolist()))
print("Missing    :", int(df.isna().sum().sum()))
print("Duplicates :", int(X.duplicated().sum()))
print("G1 DATASET: PASS")

# ------------------------------------------------------------------------------
# SPLIT CONSTRUCTION
# ------------------------------------------------------------------------------

def make_split(X, y, seed):
    indices = np.arange(len(y), dtype=np.int64)
    tr, te = train_test_split(
        indices,
        test_size=0.20,
        stratify=y,
        random_state=seed,
    )
    tr = np.asarray(tr, dtype=np.int64)
    te = np.asarray(te, dtype=np.int64)

    # Exact persisted split identity.
    h = hashlib.sha256()
    h.update(tr.tobytes())
    h.update(te.tobytes())

    return tr, te, h.hexdigest()

split_records = {}
split_arrays = {}

for seed in SEEDS:
    tr, te, h = make_split(X, y, seed)
    split_records[str(seed)] = {
        "seed": seed,
        "train_rows": int(len(tr)),
        "test_rows": int(len(te)),
        "split_hash": h,
        "archived_prefix": ARCHIVED_PREFIXES[seed],
        "archived_prefix_match": h.startswith(ARCHIVED_PREFIXES[seed]),
    }
    split_arrays[f"seed_{seed}_train"] = tr
    split_arrays[f"seed_{seed}_test"] = te

np.savez_compressed(SPLITS, **split_arrays)

print()
print("SPLIT VALIDATION")
print("-" * 78)
for seed in SEEDS:
    r = split_records[str(seed)]
    print(
        f"seed={seed:<4} train={r['train_rows']:5d} "
        f"test={r['test_rows']:4d} "
        f"new_hash={r['split_hash']} "
        f"archived_prefix={r['archived_prefix']} "
        f"archived_prefix_match={r['archived_prefix_match']}"
    )

# The old prefixes are deliberately NOT required to match.
# They document the irrecoverable historical split identity.
assert all(
    split_records[str(s)]["train_rows"] == 36168 and
    split_records[str(s)]["test_rows"] == 9043
    for s in SEEDS
)

# Re-load and verify exact persisted arrays.
saved = np.load(SPLITS)
for seed in SEEDS:
    tr1 = split_arrays[f"seed_{seed}_train"]
    te1 = split_arrays[f"seed_{seed}_test"]
    tr2 = saved[f"seed_{seed}_train"]
    te2 = saved[f"seed_{seed}_test"]
    assert np.array_equal(tr1, tr2)
    assert np.array_equal(te1, te2)

print("G2 EXACT SPLIT PERSISTENCE: PASS")

# ------------------------------------------------------------------------------
# PREPROCESSING
# ------------------------------------------------------------------------------

NUMERIC = [
    "age", "balance", "day", "duration",
    "campaign", "pdays", "previous"
]

CATEGORICAL = [
    "job", "marital", "education", "default",
    "housing", "loan", "contact", "month", "poutcome"
]

assert set(NUMERIC + CATEGORICAL) == set(X.columns)

def make_preprocessor(scaling):
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scaling:
        num_steps.append(("scaler", StandardScaler()))

    cat_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        ),
    ]

    return ColumnTransformer(
        [
            ("num", Pipeline(num_steps), NUMERIC),
            ("cat", Pipeline(cat_steps), CATEGORICAL),
        ],
        remainder="drop",
    )

# ------------------------------------------------------------------------------
# MODEL FACTORY
# ------------------------------------------------------------------------------

def model_factory(name, seed):
    if name == "logistic_regression":
        return LogisticRegression(
            solver="liblinear",
            max_iter=2000,
            random_state=seed,
        )

    if name == "decision_tree":
        return DecisionTreeClassifier(
            random_state=seed,
        )

    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            random_state=seed,
            n_jobs=-1,
        )

    if name == "linear_svm":
        return LinearSVC(
            C=1.0,
            loss="squared_hinge",
            dual=False,
            tol=1e-3,
            max_iter=1000,
            random_state=seed,
        )

    raise ValueError(name)

# ------------------------------------------------------------------------------
# DETERMINISTIC OVERSAMPLING
# ------------------------------------------------------------------------------

def deterministic_oversample(A, y_train, seed):
    y_train = np.asarray(y_train)
    classes, counts = np.unique(y_train, return_counts=True)
    target_n = int(np.max(counts))

    rng = np.random.RandomState(seed)
    parts = []

    for cls, count in zip(classes, counts):
        idx = np.flatnonzero(y_train == cls)

        if count < target_n:
            extra = rng.choice(
                idx,
                size=target_n - int(count),
                replace=True,
            )
            chosen = np.concatenate([idx, extra])
        else:
            chosen = idx.copy()

        parts.append(chosen)

    combined = np.concatenate(parts)

    # Deterministic final shuffle.
    rng2 = np.random.RandomState(seed + 1000003)
    rng2.shuffle(combined)

    return A[combined], y_train[combined]

# ------------------------------------------------------------------------------
# EXPERIMENT ENGINE
# ------------------------------------------------------------------------------

def run_one(dataset_name, model_name, seed, config_name):
    cfg = CONFIGS[config_name]

    tr_idx, te_idx, split_hash = make_split(X, y, seed)

    X_train = X.iloc[tr_idx].copy()
    X_test = X.iloc[te_idx].copy()
    y_train = y.iloc[tr_idx].to_numpy()
    y_test = y.iloc[te_idx].to_numpy()

    if cfg["training_fraction"] < 1.0:
        frac_train_idx, _ = train_test_split(
            np.arange(len(y_train)),
            train_size=cfg["training_fraction"],
            stratify=y_train,
            random_state=seed,
        )
        frac_train_idx = np.asarray(frac_train_idx, dtype=np.int64)
        X_train = X_train.iloc[frac_train_idx].copy()
        y_train = y_train[frac_train_idx]

    t0 = time.time()

    pre = make_preprocessor(cfg["scaling"])

    A = pre.fit_transform(X_train, y_train)
    B = pre.transform(X_test)

    if cfg["representation"] == "pca95":
        pca = PCA(n_components=0.95, svd_solver="full")
        A = pca.fit_transform(A)
        B = pca.transform(B)

    if cfg["sampling"] == "minority_oversampling":
        A, y_train = deterministic_oversample(A, y_train, seed)

    model = model_factory(model_name, seed)

    with warnings.catch_warnings(record=True) as ww:
        warnings.simplefilter("always")
        model.fit(A, y_train)

    pred = model.predict(B)

    elapsed = time.time() - t0

    mf1 = float(f1_score(y_test, pred, average="macro"))
    acc = float(accuracy_score(y_test, pred))
    ba = float(balanced_accuracy_score(y_test, pred))

    assert np.isfinite(mf1)
    assert np.isfinite(acc)
    assert np.isfinite(ba)

    return {
        "run_key":
            f"{dataset_name}__{model_name}__seed{seed}__{config_name}",
        "dataset": dataset_name,
        "model": model_name,
        "seed": seed,
        "configuration": config_name,
        "scaling": cfg["scaling"],
        "sampling": cfg["sampling"],
        "representation": cfg["representation"],
        "training_fraction": cfg["training_fraction"],
        "split_hash": split_hash,
        "train_rows_after_fraction": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "macro_f1": mf1,
        "accuracy": acc,
        "balanced_accuracy": ba,
        "runtime_seconds": float(elapsed),
        "warnings": json.dumps([str(w.message) for w in ww]),
        "status": "PASS",
        "error": "",
        "completed_at": datetime.now().isoformat(),
    }

# ------------------------------------------------------------------------------
# VALIDATION RUNS — 40 RUNS
# ------------------------------------------------------------------------------

rows = []

print()
print("=" * 78)
print("VALIDATION RUNS")
print("=" * 78)
print("Runs:", EXPECTED_RUNS)

for seed in SEEDS:
    for model in MODELS:
        for cfg in CONFIGS:
            print(
                f"RUN | seed={seed:<4} | "
                f"{model:<22} | {cfg}"
            )

            try:
                row = run_one(
                    "bank_marketing",
                    model,
                    seed,
                    cfg,
                )
                rows.append(row)

                print(
                    f"PASS | Macro-F1={row['macro_f1']:.6f} | "
                    f"BA={row['balanced_accuracy']:.6f} | "
                    f"time={row['runtime_seconds']:.2f}s"
                )

            except Exception as exc:
                rows.append({
                    "run_key":
                        f"bank_marketing__{model}__seed{seed}__{cfg}",
                    "dataset": "bank_marketing",
                    "model": model,
                    "seed": seed,
                    "configuration": cfg,
                    "status": "FAIL",
                    "error": repr(exc),
                })
                print("FAIL:", repr(exc))
                raise

            gc.collect()

results = pd.DataFrame(rows)
results.to_csv(RESULTS, index=False)

# ------------------------------------------------------------------------------
# STRUCTURAL GATES
# ------------------------------------------------------------------------------

print()
print("=" * 78)
print("VALIDATION GATES")
print("=" * 78)

assert len(results) == EXPECTED_RUNS
assert results["run_key"].nunique() == EXPECTED_RUNS
assert (results["status"] == "PASS").all()

for cfg in CONFIGS:
    assert int((results["configuration"] == cfg).sum()) == 8

for model in MODELS:
    assert int((results["model"] == model).sum()) == 10

for seed in SEEDS:
    q = results[results["seed"] == seed]
    assert len(q) == 20
    assert q["split_hash"].nunique() == 1
    assert q["test_rows"].nunique() == 1

assert results["macro_f1"].notna().all()
assert results["balanced_accuracy"].notna().all()
assert results["accuracy"].notna().all()
assert (results["runtime_seconds"] > 0).all()

print("G3 PREPROCESSING / CONFIGURATION PATHS: PASS")
print("G4 MODEL × CONFIGURATION COVERAGE: PASS")
print("G6 METRIC INTEGRITY: PASS")
print("G7 RESOURCE FEASIBILITY: PASS")

# ------------------------------------------------------------------------------
# DETERMINISTIC REPEAT
# ------------------------------------------------------------------------------

print()
print("=" * 78)
print("DETERMINISTIC REPEAT")
print("=" * 78)

repeat_cases = [
    (42, "logistic_regression", "baseline"),
    (42, "logistic_regression", "sampling_cf"),
    (123, "random_forest", "baseline"),
]

repeat_results = []

for seed, model, cfg in repeat_cases:
    a = run_one("bank_marketing", model, seed, cfg)
    b = run_one("bank_marketing", model, seed, cfg)

    repeat_results.append({
        "seed": seed,
        "model": model,
        "configuration": cfg,
        "macro_f1_a": a["macro_f1"],
        "macro_f1_b": b["macro_f1"],
        "balanced_accuracy_a": a["balanced_accuracy"],
        "balanced_accuracy_b": b["balanced_accuracy"],
        "exact_macro_f1_match":
            a["macro_f1"] == b["macro_f1"],
        "exact_ba_match":
            a["balanced_accuracy"] == b["balanced_accuracy"],
        "exact_split_match":
            a["split_hash"] == b["split_hash"],
    })

repeat_df = pd.DataFrame(repeat_results)

assert repeat_df["exact_macro_f1_match"].all()
assert repeat_df["exact_ba_match"].all()
assert repeat_df["exact_split_match"].all()

print(repeat_df.to_string(index=False))
print("G5 DETERMINISTIC REPEAT: PASS")

# ------------------------------------------------------------------------------
# ARCHIVED COMPARISON — INFORMATIONAL ONLY
# ------------------------------------------------------------------------------

print()
print("=" * 78)
print("ARCHIVED BATCH 1B REPRESENTATIVE COMPARISON")
print("=" * 78)
print("NOTE: comparison only; NOT an equivalence test.")

comparison = []

for key, archived in ARCHIVED_REPRESENTATIVES.items():
    seed, model, cfg = key
    q = results[
        (results["seed"] == seed) &
        (results["model"] == model) &
        (results["configuration"] == cfg)
    ]

    assert len(q) == 1
    current = q.iloc[0]

    comparison.append({
        "seed": seed,
        "model": model,
        "configuration": cfg,
        "archived_macro_f1": archived["macro_f1"],
        "current_macro_f1": current["macro_f1"],
        "macro_f1_difference":
            current["macro_f1"] - archived["macro_f1"],
        "archived_ba": archived["balanced_accuracy"],
        "current_ba": current["balanced_accuracy"],
        "ba_difference":
            current["balanced_accuracy"] -
            archived["balanced_accuracy"],
        "archived_split_prefix":
            ARCHIVED_PREFIXES[seed],
        "current_full_split_hash":
            current["split_hash"],
    })

comparison_df = pd.DataFrame(comparison)
print(comparison_df.to_string(index=False))

# ------------------------------------------------------------------------------
# FINAL REPORT
# ------------------------------------------------------------------------------

report = {
    "type": "validation_only",
    "dataset": "bank_marketing",
    "dataset_path": str(data_path),
    "dataset_shape": list(df.shape),
    "expected_runs": EXPECTED_RUNS,
    "completed_runs": int(len(results)),
    "pass_runs": int((results["status"] == "PASS").sum()),
    "fail_runs": int((results["status"] != "PASS").sum()),
    "models": MODELS,
    "seeds": SEEDS,
    "configurations": list(CONFIGS),
    "primary_metric": "macro_f1",
    "split_policy": "stratified 80/20; exact indices persisted",
    "split_records": split_records,
    "persisted_split_file": str(SPLITS),
    "results_file": str(RESULTS),
    "deterministic_repeat": repeat_results,
    "archived_comparison": comparison,
    "archived_prefixes_are_not_reproduced":
        not all(r["archived_prefix_match"] for r in split_records.values()),
    "production_authorized": False,
    "created_at": datetime.now().isoformat(),
}

with open(REPORT, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

manifest = {
    "purpose":
        "Bank Marketing evidence-recovery validation; production not authorized",
    "dataset": "bank_marketing",
    "dataset_path": str(data_path),
    "models": MODELS,
    "seeds": SEEDS,
    "configurations": CONFIGS,
    "expected_runs": EXPECTED_RUNS,
    "results": str(RESULTS),
    "splits": str(SPLITS),
    "report": str(REPORT),
    "historical_batch1b_prefixes": ARCHIVED_PREFIXES,
    "historical_prefix_status":
        "preserved as provenance only; not reproduced by current split reconstruction",
}

with open(MANIFEST, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print()
print("=" * 78)
print("BATCH 1B-R1 VALIDATION COMPLETE")
print("=" * 78)
print("Results :", RESULTS)
print("Splits  :", SPLITS)
print("Report  :", REPORT)
print("Manifest:", MANIFEST)
print()
print("PRODUCTION AUTHORIZATION: NO")
print("The next decision must be based on this validation output.")