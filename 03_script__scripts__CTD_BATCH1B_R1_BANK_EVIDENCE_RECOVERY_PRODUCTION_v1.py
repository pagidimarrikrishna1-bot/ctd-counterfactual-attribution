# ==============================================================================
# CTD IJDSA REVISION
# BATCH 1B-R1 — BANK MARKETING EVIDENCE RECOVERY PRODUCTION v1.0
#
# PURPOSE
# -------
# Produce a NEW, fully reproducible Bank Marketing evidence block.
#
# IMPORTANT
# ---------
# This does NOT modify or overwrite the historical Batch 1B ledger.
# The historical Batch 1B evidence remains archived as recovered evidence.
#
# Frozen scientific design:
#   Dataset: Bank Marketing, validated UCI bank-full.csv
#   Seeds: 15 frozen seeds
#   Models: 4
#   Configurations: 5
#   Total production fits: 300
#   Primary metric: Macro-F1
#   Split: stratified 80/20, seed-specific
#
# Every seed's exact train/test indices and FULL SHA-256 split hash are
# persisted in the production split registry.
#
# Production is authorized only because Batch 1B-R1 validation has already
# passed:
#   40/40 PASS
#   dataset gate PASS
#   split persistence PASS
#   preprocessing/configuration paths PASS
#   model × configuration coverage PASS
#   metric integrity PASS
#   deterministic repeat PASS
#   resource feasibility PASS
#
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

BASE = Path("/content/CTD_BATCH1B_R1_PRODUCTION")
BASE.mkdir(parents=True, exist_ok=True)

SEEDS = [
    42, 123, 2026, 7, 19,
    31, 53, 71, 89, 107,
    149, 173, 211, 257, 311
]

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

EXPECTED = len(SEEDS) * len(MODELS) * len(CONFIGS)

RESULTS = BASE / "CTD_BATCH1B_R1_RESULTS_LEDGER.csv"
STATUS = BASE / "CTD_BATCH1B_R1_RUN_STATUS.csv"
SPLIT_REGISTRY = BASE / "CTD_BATCH1B_R1_EXACT_SPLIT_REGISTRY.npz"
SPLIT_MANIFEST = BASE / "CTD_BATCH1B_R1_SPLIT_MANIFEST.json"
MANIFEST = BASE / "CTD_BATCH1B_R1_PRODUCTION_MANIFEST.json"
CHECKPOINT = BASE / "CTD_BATCH1B_R1_CHECKPOINT.json"
FINAL = BASE / "CTD_BATCH1B_R1_FINAL_VALIDATION_REPORT.json"


# ------------------------------------------------------------------------------
# DATASET
# ------------------------------------------------------------------------------

DATASET_CANDIDATES = [
    Path("/content/bank-full.csv"),
    Path("/content/Bank Marketing/bank-full.csv"),
]

DATASET_PATH = next(
    (p for p in DATASET_CANDIDATES if p.exists()),
    None
)

if DATASET_PATH is None:
    raise FileNotFoundError(
        "Validated bank-full.csv not found in /content."
    )

df = pd.read_csv(DATASET_PATH, sep=";")

assert df.shape == (45211, 17), df.shape

EXPECTED_COLUMNS = [
    "age", "job", "marital", "education", "default",
    "balance", "housing", "loan", "contact", "day",
    "month", "duration", "campaign", "pdays",
    "previous", "poutcome", "y"
]

assert list(df.columns) == EXPECTED_COLUMNS
assert int(df.isna().sum().sum()) == 0

X = df.drop(columns=["y"]).copy()
y = (
    df["y"]
    .astype(str)
    .str.strip()
    .map({"no": 0, "yes": 1})
    .astype(int)
)

assert sorted(y.unique().tolist()) == [0, 1]
assert int(X.duplicated().sum()) == 0


# ------------------------------------------------------------------------------
# EXACT SPLITS
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

    h = hashlib.sha256()
    h.update(tr.tobytes())
    h.update(te.tobytes())

    return tr, te, h.hexdigest()


split_records = {}
split_arrays = {}

for seed in SEEDS:
    tr, te, split_hash = make_split(X, y, seed)

    split_records[str(seed)] = {
        "seed": seed,
        "train_rows": int(len(tr)),
        "test_rows": int(len(te)),
        "split_hash": split_hash,
    }

    split_arrays[f"seed_{seed}_train"] = tr
    split_arrays[f"seed_{seed}_test"] = te


# Save exact arrays before production.
np.savez_compressed(
    SPLIT_REGISTRY,
    **split_arrays
)

with open(SPLIT_MANIFEST, "w", encoding="utf-8") as f:
    json.dump(
        {
            "dataset": "bank_marketing",
            "dataset_path": str(DATASET_PATH),
            "dataset_shape": list(df.shape),
            "split_policy": "stratified 80/20",
            "exact_split_arrays_file": str(SPLIT_REGISTRY),
            "splits": split_records,
            "created_at": datetime.now().isoformat(),
        },
        f,
        indent=2,
    )


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
    num_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if scaling:
        num_steps.append(
            ("scaler", StandardScaler())
        )

    cat_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
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

    classes, counts = np.unique(
        y_train,
        return_counts=True
    )

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

            chosen = np.concatenate(
                [idx, extra]
            )
        else:
            chosen = idx.copy()

        parts.append(chosen)

    combined = np.concatenate(parts)

    rng2 = np.random.RandomState(
        seed + 1000003
    )

    rng2.shuffle(combined)

    return (
        A[combined],
        y_train[combined]
    )


# ------------------------------------------------------------------------------
# SINGLE EXPERIMENT
# ------------------------------------------------------------------------------

def run_one(
    model_name,
    seed,
    config_name,
    split_registry,
):

    cfg = CONFIGS[config_name]

    tr_idx = split_registry[
        f"seed_{seed}_train"
    ]

    te_idx = split_registry[
        f"seed_{seed}_test"
    ]

    split_hash = split_records[
        str(seed)
    ]["split_hash"]

    X_train = X.iloc[tr_idx].copy()
    X_test = X.iloc[te_idx].copy()

    y_train = y.iloc[tr_idx].to_numpy()
    y_test = y.iloc[te_idx].to_numpy()

    if cfg["training_fraction"] < 1.0:

        frac_idx, _ = train_test_split(
            np.arange(len(y_train)),
            train_size=cfg["training_fraction"],
            stratify=y_train,
            random_state=seed,
        )

        frac_idx = np.asarray(
            frac_idx,
            dtype=np.int64
        )

        X_train = X_train.iloc[
            frac_idx
        ].copy()

        y_train = y_train[
            frac_idx
        ]

    t0 = time.time()

    pre = make_preprocessor(
        cfg["scaling"]
    )

    A = pre.fit_transform(
        X_train,
        y_train
    )

    B = pre.transform(
        X_test
    )

    if cfg["representation"] == "pca95":

        pca = PCA(
            n_components=0.95,
            svd_solver="full"
        )

        A = pca.fit_transform(A)
        B = pca.transform(B)

    if cfg["sampling"] == "minority_oversampling":

        A, y_train = deterministic_oversample(
            A,
            y_train,
            seed
        )

    model = model_factory(
        model_name,
        seed
    )

    with warnings.catch_warnings(
        record=True
    ) as ww:

        warnings.simplefilter("always")

        model.fit(
            A,
            y_train
        )

    pred = model.predict(B)

    elapsed = time.time() - t0

    mf1 = float(
        f1_score(
            y_test,
            pred,
            average="macro"
        )
    )

    acc = float(
        accuracy_score(
            y_test,
            pred
        )
    )

    ba = float(
        balanced_accuracy_score(
            y_test,
            pred
        )
    )

    assert np.isfinite(mf1)
    assert np.isfinite(acc)
    assert np.isfinite(ba)

    return {
        "run_key":
            f"bank_marketing__{model_name}"
            f"__seed{seed}__{config_name}",

        "dataset":
            "bank_marketing",

        "model":
            model_name,

        "seed":
            seed,

        "configuration":
            config_name,

        "scaling":
            cfg["scaling"],

        "sampling":
            cfg["sampling"],

        "representation":
            cfg["representation"],

        "training_fraction":
            cfg["training_fraction"],

        "split_hash":
            split_hash,

        "train_rows_after_fraction":
            int(len(y_train)),

        "test_rows":
            int(len(y_test)),

        "macro_f1":
            mf1,

        "accuracy":
            acc,

        "balanced_accuracy":
            ba,

        "runtime_seconds":
            float(elapsed),

        "warnings":
            json.dumps(
                [
                    str(w.message)
                    for w in ww
                ]
            ),

        "status":
            "PASS",

        "error":
            "",

        "completed_at":
            datetime.now().isoformat(),
    }


# ------------------------------------------------------------------------------
# CHECKPOINT / RESUME
# ------------------------------------------------------------------------------

columns = [
    "run_key",
    "dataset",
    "model",
    "seed",
    "configuration",
    "scaling",
    "sampling",
    "representation",
    "training_fraction",
    "split_hash",
    "train_rows_after_fraction",
    "test_rows",
    "macro_f1",
    "accuracy",
    "balanced_accuracy",
    "runtime_seconds",
    "warnings",
    "status",
    "error",
    "completed_at",
]

if RESULTS.exists():
    existing = pd.read_csv(RESULTS)

    if "run_key" in existing.columns:
        completed_keys = set(
            existing.loc[
                existing["status"].astype(str).eq("PASS"),
                "run_key"
            ].astype(str)
        )
    else:
        existing = pd.DataFrame(
            columns=columns
        )
        completed_keys = set()

else:
    existing = pd.DataFrame(
        columns=columns
    )
    completed_keys = set()


# ------------------------------------------------------------------------------
# MANIFEST
# ------------------------------------------------------------------------------

manifest_rows = []

for seed in SEEDS:
    for model in MODELS:
        for cfg in CONFIGS:

            manifest_rows.append(
                {
                    "run_key":
                        f"bank_marketing__{model}"
                        f"__seed{seed}__{cfg}",

                    "dataset":
                        "bank_marketing",

                    "model":
                        model,

                    "seed":
                        seed,

                    "configuration":
                        cfg,
                }
            )

manifest_df = pd.DataFrame(
    manifest_rows
)

assert len(manifest_df) == EXPECTED
assert manifest_df["run_key"].nunique() == EXPECTED


# ------------------------------------------------------------------------------
# STATUS FILE
# ------------------------------------------------------------------------------

status_rows = []

for row in manifest_rows:

    status_rows.append(
        {
            **row,
            "status":
                "PASS"
                if row["run_key"]
                in completed_keys
                else "PENDING",
        }
    )

pd.DataFrame(
    status_rows
).to_csv(
    STATUS,
    index=False
)


# ------------------------------------------------------------------------------
# PRODUCTION HEADER
# ------------------------------------------------------------------------------

print("=" * 78)
print("CTD IJDSA REVISION")
print("BATCH 1B-R1 — BANK MARKETING EVIDENCE RECOVERY PRODUCTION v1.0")
print("=" * 78)

print("Python       :", platform.python_version())
print("NumPy        :", np.__version__)
print("Pandas       :", pd.__version__)
print("Scikit-learn:", sklearn.__version__)

print()
print("Dataset      :", DATASET_PATH)
print("Shape        :", df.shape)
print("Expected fits:", EXPECTED)
print("Already PASS :", len(completed_keys))
print("Remaining    :", EXPECTED - len(completed_keys))

print()
print("Seeds        :", SEEDS)
print("Models       :", MODELS)
print("Configurations:", list(CONFIGS))

print()
print("EXACT SPLIT REGISTRY")
for seed in SEEDS:
    print(
        f"seed={seed:<4} "
        f"train={split_records[str(seed)]['train_rows']} "
        f"test={split_records[str(seed)]['test_rows']} "
        f"hash={split_records[str(seed)]['split_hash']}"
    )

print()
print("Historical Batch 1B evidence is NOT modified.")
print("This is a new reproducible evidence block.")
print("=" * 78)


# ------------------------------------------------------------------------------
# PRODUCTION LOOP
# ------------------------------------------------------------------------------

all_rows = []

if len(existing):
    all_rows.extend(
        existing.to_dict(
            orient="records"
        )
    )

for seed in SEEDS:

    for model in MODELS:

        for cfg in CONFIGS:

            run_key = (
                f"bank_marketing__{model}"
                f"__seed{seed}__{cfg}"
            )

            if run_key in completed_keys:
                print(
                    "SKIP existing:",
                    run_key
                )
                continue

            print()
            print(
                f"RUN | seed={seed:<4} | "
                f"{model:<22} | {cfg}"
            )

            try:

                row = run_one(
                    model,
                    seed,
                    cfg,
                    split_arrays
                )

                all_rows.append(row)

                results_df = pd.DataFrame(
                    all_rows
                )

                results_df.to_csv(
                    RESULTS,
                    index=False
                )

                # Update checkpoint.
                completed_keys.add(
                    run_key
                )

                with open(
                    CHECKPOINT,
                    "w",
                    encoding="utf-8"
                ) as f:

                    json.dump(
                        {
                            "dataset":
                                "bank_marketing",

                            "expected_runs":
                                EXPECTED,

                            "completed_pass":
                                len(completed_keys),

                            "remaining":
                                EXPECTED -
                                len(completed_keys),

                            "updated_at":
                                datetime.now().isoformat(),
                        },
                        f,
                        indent=2
                    )

                print(
                    f"PASS | "
                    f"Macro-F1={row['macro_f1']:.6f} | "
                    f"BA={row['balanced_accuracy']:.6f} | "
                    f"time={row['runtime_seconds']:.2f}s"
                )

            except Exception as exc:

                error_row = {
                    "run_key": run_key,
                    "dataset": "bank_marketing",
                    "model": model,
                    "seed": seed,
                    "configuration": cfg,
                    "status": "FAIL",
                    "error": repr(exc),
                }

                all_rows.append(
                    error_row
                )

                pd.DataFrame(
                    all_rows
                ).to_csv(
                    RESULTS,
                    index=False
                )

                raise

            gc.collect()


# ------------------------------------------------------------------------------
# FINAL VALIDATION
# ------------------------------------------------------------------------------

results = pd.read_csv(RESULTS)

assert len(results) == EXPECTED
assert results["run_key"].nunique() == EXPECTED

assert (
    results["status"]
    .astype(str)
    .eq("PASS")
    .all()
)

assert results["dataset"].astype(str).eq(
    "bank_marketing"
).all()

assert set(
    results["model"].astype(str)
) == set(MODELS)

assert set(
    results["configuration"].astype(str)
) == set(CONFIGS)

assert set(
    results["seed"].astype(int)
) == set(SEEDS)

assert results["macro_f1"].notna().all()
assert results["accuracy"].notna().all()
assert results["balanced_accuracy"].notna().all()
assert results["runtime_seconds"].gt(0).all()

# Exactly 20 runs per seed/split.
for seed in SEEDS:

    q = results[
        results["seed"].astype(int) == seed
    ]

    assert len(q) == 20
    assert q["split_hash"].nunique() == 1

# Exactly 60 per configuration.
for cfg in CONFIGS:

    q = results[
        results["configuration"].astype(str)
        == cfg
    ]

    assert len(q) == 60

# Exactly 75 per model.
for model in MODELS:

    q = results[
        results["model"].astype(str)
        == model
    ]

    assert len(q) == 75

# Verify exact split registry.
saved = np.load(
    SPLIT_REGISTRY
)

for seed in SEEDS:

    tr_saved = saved[
        f"seed_{seed}_train"
    ]

    te_saved = saved[
        f"seed_{seed}_test"
    ]

    tr_current = split_arrays[
        f"seed_{seed}_train"
    ]

    te_current = split_arrays[
        f"seed_{seed}_test"
    ]

    assert np.array_equal(
        tr_saved,
        tr_current
    )

    assert np.array_equal(
        te_saved,
        te_current
    )

# Verify split hash consistency.
for seed in SEEDS:

    expected_hash = split_records[
        str(seed)
    ]["split_hash"]

    q = results[
        results["seed"].astype(int) == seed
    ]

    assert q["split_hash"].nunique() == 1
    assert (
        q["split_hash"].iloc[0]
        == expected_hash
    )


# ------------------------------------------------------------------------------
# FINAL REPORT
# ------------------------------------------------------------------------------

final_report = {
    "dataset":
        "bank_marketing",

    "dataset_path":
        str(DATASET_PATH),

    "dataset_shape":
        list(df.shape),

    "expected_runs":
        EXPECTED,

    "completed_runs":
        int(len(results)),

    "pass_runs":
        int(
            results["status"]
            .astype(str)
            .eq("PASS")
            .sum()
        ),

    "fail_runs":
        int(
            (~results["status"]
             .astype(str)
             .eq("PASS"))
            .sum()
        ),

    "models":
        MODELS,

    "seeds":
        SEEDS,

    "configurations":
        list(CONFIGS),

    "primary_metric":
        "macro_f1",

    "split_policy":
        "stratified 80/20",

    "exact_split_registry":
        str(SPLIT_REGISTRY),

    "split_manifest":
        str(SPLIT_MANIFEST),

    "historical_batch1b_status":
        "preserved separately; not overwritten",

    "reproducibility_status":
        "FULL_EXACT_SPLIT_REGISTRY",

    "validation_status":
        "PASS",

    "production_status":
        "COMPLETE",

    "created_at":
        datetime.now().isoformat(),
}

with open(
    FINAL,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        final_report,
        f,
        indent=2
    )

production_manifest = {
    "purpose":
        "New reproducible Bank Marketing CTD evidence block",

    "dataset":
        "bank_marketing",

    "dataset_path":
        str(DATASET_PATH),

    "models":
        MODELS,

    "seeds":
        SEEDS,

    "configurations":
        CONFIGS,

    "expected_runs":
        EXPECTED,

    "results":
        str(RESULTS),

    "status":
        str(STATUS),

    "exact_split_registry":
        str(SPLIT_REGISTRY),

    "split_manifest":
        str(SPLIT_MANIFEST),

    "checkpoint":
        str(CHECKPOINT),

    "final_report":
        str(FINAL),

    "historical_ledger_untouched":
        True,

    "created_at":
        datetime.now().isoformat(),
}

with open(
    MANIFEST,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        production_manifest,
        f,
        indent=2
    )


print()
print("=" * 78)
print("BATCH 1B-R1 PRODUCTION COMPLETE")
print("=" * 78)

print("Expected runs :", EXPECTED)
print("Completed     :", len(results))
print(
    "PASS          :",
    int(
        results["status"]
        .astype(str)
        .eq("PASS")
        .sum()
    )
)
print(
    "FAIL          :",
    int(
        (~results["status"]
         .astype(str)
         .eq("PASS"))
        .sum()
    )
)

print()
print("Results ledger :", RESULTS)
print("Status         :", STATUS)
print("Exact splits   :", SPLIT_REGISTRY)
print("Split manifest :", SPLIT_MANIFEST)
print("Checkpoint     :", CHECKPOINT)
print("Manifest       :", MANIFEST)
print("Final report   :", FINAL)

print()
print("HISTORICAL BATCH 1B LEDGER: UNTOUCHED")
print("NEW BATCH 1B-R1 EVIDENCE: FULLY REPRODUCIBLE")
print("=" * 78)