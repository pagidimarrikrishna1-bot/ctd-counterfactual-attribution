
# ================================================================
# CTD IJDSA REVISION — BATCH 2 ENGINE VALIDATION v1
# PIMA + SPAMBASE + PHONEME
# ================================================================
#
# PURPOSE:
#   Validate the CTD engine on the three planned new datasets
#   BEFORE any production campaign.
#
# DESIGN:
#   3 datasets × 4 models × 5 configurations × 2 seeds = 120 runs
#
# IMPORTANT:
#   - VALIDATION ONLY
#   - NO 900-run production
#   - NO interaction experiments
#   - NO changes to the frozen CTD scientific protocol
#
# DATASETS:
#   1. Pima Indians Diabetes
#   2. Spambase
#   3. Phoneme
#
# MODELS:
#   Logistic Regression
#   Decision Tree
#   Random Forest
#   Linear SVM
#
# CONFIGURATIONS:
#   baseline
#   scaling_cf
#   sampling_cf
#   representation_cf
#   training_fraction_cf
#
# SEEDS:
#   42, 123
#
# PRIMARY METRIC:
#   Macro-F1
#
# PRODUCTION AUTHORIZATION:
#   ALWAYS FALSE
# ================================================================

import os
import gc
import json
import time
import hashlib
import platform
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    balanced_accuracy_score
)


# ================================================================
# 0. FROZEN SETTINGS
# ================================================================

BASE = Path(
    "/content/CTD_BATCH2_ENGINE_VALIDATION"
)

BASE.mkdir(
    parents=True,
    exist_ok=True
)

SEEDS = [42, 123]

DATASETS = [
    "pima_diabetes",
    "spambase",
    "phoneme"
]

MODELS = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "linear_svm"
]

CONFIGURATIONS = {

    "baseline": {
        "scaling": True,
        "sampling": "original",
        "representation": "all_features",
        "training_fraction": 1.0
    },

    "scaling_cf": {
        "scaling": False,
        "sampling": "original",
        "representation": "all_features",
        "training_fraction": 1.0
    },

    "sampling_cf": {
        "scaling": True,
        "sampling": "minority_oversampling",
        "representation": "all_features",
        "training_fraction": 1.0
    },

    "representation_cf": {
        "scaling": True,
        "sampling": "original",
        "representation": "pca95",
        "training_fraction": 1.0
    },

    "training_fraction_cf": {
        "scaling": True,
        "sampling": "original",
        "representation": "all_features",
        "training_fraction": 0.5
    }
}

EXPECTED_RUNS = (
    len(DATASETS)
    * len(MODELS)
    * len(CONFIGURATIONS)
    * len(SEEDS)
)

print("=" * 78)
print("CTD IJDSA REVISION — BATCH 2 ENGINE VALIDATION v1")
print("=" * 78)

print("\n[0] FROZEN DESIGN")
print("-" * 78)

print("Datasets        :", DATASETS)
print("Models          :", MODELS)
print("Configurations :", list(CONFIGURATIONS.keys()))
print("Seeds           :", SEEDS)
print("Expected runs   :", EXPECTED_RUNS)
print("Primary metric  : Macro-F1")
print("Production      : FALSE")


# ================================================================
# 1. ENVIRONMENT
# ================================================================

print("\n[1] ENVIRONMENT")
print("-" * 78)

print("Python      :", platform.python_version())
print("NumPy       :", np.__version__)
print("Pandas      :", pd.__version__)
print("Scikit-learn:", sklearn.__version__)


# ================================================================
# 2. DATASET LOADERS
# ================================================================

def load_pima():

    # OpenML Pima Indians Diabetes.
    # Search several known identifiers without silently changing
    # the dataset if a different source is returned.
    candidates = [
        ("diabetes", 37),
        ("diabetes", None)
    ]

    last_error = None

    for name, did in candidates:

        try:

            if did is None:
                ds = fetch_openml(
                    name=name,
                    as_frame=True
                )
            else:
                ds = fetch_openml(
                    data_id=did,
                    as_frame=True
                )

            X = ds.data.copy()
            y = ds.target.copy()

            return X, y, {
                "source": "OpenML",
                "name": name,
                "data_id": did
            }

        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Pima loading failed: {last_error}"
    )


def load_spambase():

    ds = fetch_openml(
        name="spambase",
        as_frame=True
    )

    return (
        ds.data.copy(),
        ds.target.copy(),
        {
            "source": "OpenML",
            "name": "spambase"
        }
    )


def load_phoneme():

    ds = fetch_openml(
        name="phoneme",
        as_frame=True
    )

    return (
        ds.data.copy(),
        ds.target.copy(),
        {
            "source": "OpenML",
            "name": "phoneme"
        }
    )


LOADERS = {
    "pima_diabetes": load_pima,
    "spambase": load_spambase,
    "phoneme": load_phoneme
}


# ================================================================
# 3. CLEAN / STANDARDIZE TARGETS
# ================================================================

def prepare_xy(X, y):

    X = pd.DataFrame(X).copy()
    y = pd.Series(y).copy()

    # Convert feature columns to numeric where possible.
    # Any conversion failure becomes NaN and is checked below.
    for c in X.columns:
        X[c] = pd.to_numeric(
            X[c],
            errors="coerce"
        )

    # Target as string for stable classification handling.
    y = y.astype(str)

    # Drop rows only if source contains missing target/features.
    # Record this explicitly rather than silently imputing.
    valid_mask = (
        ~X.isna().any(axis=1)
        &
        ~y.isna()
    )

    removed = int(
        (~valid_mask).sum()
    )

    X = X.loc[
        valid_mask
    ].reset_index(drop=True)

    y = y.loc[
        valid_mask
    ].reset_index(drop=True)

    return X, y, removed


# ================================================================
# 4. DETERMINISTIC MULTICLASS / BINARY OVERSAMPLING
# ================================================================

def deterministic_random_oversample(
    X_train,
    y_train,
    seed
):

    rng = np.random.RandomState(
        seed
    )

    y_array = np.asarray(
        y_train
    )

    classes, counts = np.unique(
        y_array,
        return_counts=True
    )

    max_count = int(
        counts.max()
    )

    parts_X = [
        np.asarray(X_train)
    ]

    parts_y = [
        y_array
    ]

    for cls, count in zip(
        classes,
        counts
    ):

        deficit = (
            max_count
            - int(count)
        )

        if deficit <= 0:
            continue

        idx = np.where(
            y_array == cls
        )[0]

        sampled = rng.choice(
            idx,
            size=deficit,
            replace=True
        )

        parts_X.append(
            np.asarray(X_train)[
                sampled
            ]
        )

        parts_y.append(
            y_array[
                sampled
            ]
        )

    X_out = np.vstack(
        parts_X
    )

    y_out = np.concatenate(
        parts_y
    )

    # Deterministically shuffle the expanded training set.
    order = rng.permutation(
        len(y_out)
    )

    return (
        X_out[order],
        y_out[order]
    )


# ================================================================
# 5. MODEL FACTORY
# ================================================================

def make_model(
    model_name,
    seed
):

    if model_name == "logistic_regression":

        return LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            random_state=seed
        )

    if model_name == "decision_tree":

        return DecisionTreeClassifier(
            random_state=seed
        )

    if model_name == "random_forest":

        return RandomForestClassifier(
            n_estimators=200,
            random_state=seed,
            n_jobs=-1
        )

    if model_name == "linear_svm":

        return LinearSVC(
            C=1.0,
            penalty="l2",
            loss="squared_hinge",
            dual=False,
            tol=1e-3,
            max_iter=1000,
            random_state=seed
        )

    raise ValueError(
        f"Unknown model: {model_name}"
    )


# ================================================================
# 6. CANONICAL SPLIT
# ================================================================

def split_hash(
    train_idx,
    test_idx
):

    h = hashlib.sha256()

    h.update(
        np.asarray(
            train_idx,
            dtype=np.int64
        ).tobytes()
    )

    h.update(
        np.asarray(
            test_idx,
            dtype=np.int64
        ).tobytes()
    )

    return h.hexdigest()


def make_split(
    X,
    y,
    seed
):

    indices = np.arange(
        len(X),
        dtype=np.int64
    )

    train_idx, test_idx = (
        train_test_split(
            indices,
            test_size=0.20,
            stratify=y,
            random_state=seed
        )
    )

    train_idx = np.asarray(
        train_idx,
        dtype=np.int64
    )

    test_idx = np.asarray(
        test_idx,
        dtype=np.int64
    )

    return (
        train_idx,
        test_idx,
        split_hash(
            train_idx,
            test_idx
        )
    )


# ================================================================
# 7. SINGLE EXPERIMENT
# ================================================================

def run_single(
    dataset_name,
    model_name,
    seed,
    configuration,
    X,
    y,
    train_idx,
    test_idx,
    split_id
):

    cfg = CONFIGURATIONS[
        configuration
    ]

    t0 = time.time()

    X_train = X.iloc[
        train_idx
    ].to_numpy(
        dtype=np.float64
    )

    X_test = X.iloc[
        test_idx
    ].to_numpy(
        dtype=np.float64
    )

    y_train = y.iloc[
        train_idx
    ].to_numpy()

    y_test = y.iloc[
        test_idx
    ].to_numpy()

    # ------------------------------------------------------------
    # Training fraction
    # ------------------------------------------------------------

    if cfg[
        "training_fraction"
    ] < 1.0:

        sub_train, _, sub_y, _ = (
            train_test_split(
                X_train,
                y_train,
                train_size=cfg[
                    "training_fraction"
                ],
                stratify=y_train,
                random_state=seed
            )
        )

        X_train = sub_train
        y_train = sub_y

    # ------------------------------------------------------------
    # Scaling
    # ------------------------------------------------------------

    if cfg["scaling"]:

        scaler = StandardScaler()

        X_train = scaler.fit_transform(
            X_train
        )

        X_test = scaler.transform(
            X_test
        )

    # ------------------------------------------------------------
    # PCA95
    # ------------------------------------------------------------

    pca_components = None
    pca_variance = None

    if (
        cfg["representation"]
        == "pca95"
    ):

        pca = PCA(
            n_components=0.95,
            svd_solver="full",
            random_state=seed
        )

        X_train = pca.fit_transform(
            X_train
        )

        X_test = pca.transform(
            X_test
        )

        pca_components = int(
            X_train.shape[1]
        )

        pca_variance = float(
            pca.explained_variance_ratio_.sum()
        )

    # ------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------

    if (
        cfg["sampling"]
        == "minority_oversampling"
    ):

        X_train, y_train = (
            deterministic_random_oversample(
                X_train,
                y_train,
                seed
            )
        )

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------

    model = make_model(
        model_name,
        seed
    )

    with warnings.catch_warnings(
        record=True
    ) as caught:

        warnings.simplefilter(
            "always"
        )

        model.fit(
            X_train,
            y_train
        )

    pred = model.predict(
        X_test
    )

    macro_f1 = float(
        f1_score(
            y_test,
            pred,
            average="macro"
        )
    )

    accuracy = float(
        accuracy_score(
            y_test,
            pred
        )
    )

    balanced_accuracy = float(
        balanced_accuracy_score(
            y_test,
            pred
        )
    )

    elapsed = (
        time.time() - t0
    )

    warning_text = [
        str(w.message)
        for w in caught
    ]

    return {

        "run_key":
            f"{dataset_name}__"
            f"{model_name}__"
            f"seed{seed}__"
            f"{configuration}",

        "dataset":
            dataset_name,

        "model":
            model_name,

        "seed":
            seed,

        "configuration":
            configuration,

        "scaling":
            cfg["scaling"],

        "sampling":
            cfg["sampling"],

        "representation":
            cfg["representation"],

        "training_fraction":
            cfg["training_fraction"],

        "split_hash":
            split_id,

        "train_rows":
            len(X_train),

        "test_rows":
            len(X_test),

        "pca_components":
            pca_components,

        "pca_explained_variance":
            pca_variance,

        "macro_f1":
            macro_f1,

        "accuracy":
            accuracy,

        "balanced_accuracy":
            balanced_accuracy,

        "runtime_seconds":
            elapsed,

        "warnings":
            json.dumps(
                warning_text
            ),

        "status":
            "PASS"
    }


# ================================================================
# 8. DATASET AUDIT
# ================================================================

dataset_audit = []

loaded_data = {}

print("\n[2] DATASET AUDIT")
print("-" * 78)

for dataset_name in DATASETS:

    print(
        f"\nLoading {dataset_name}..."
    )

    t0 = time.time()

    X_raw, y_raw, metadata = (
        LOADERS[
            dataset_name
        ]()
    )

    X, y, removed = prepare_xy(
        X_raw,
        y_raw
    )

    elapsed = (
        time.time() - t0
    )

    print(
        "Shape:",
        X.shape
    )

    print(
        "Classes:",
        y.nunique()
    )

    print(
        "Removed rows:",
        removed
    )

    print(
        "Missing values:",
        int(
            X.isna().sum().sum()
        )
    )

    print(
        "Duplicate feature rows:",
        int(
            X.duplicated().sum()
        )
    )

    assert len(X) == len(y)
    assert X.shape[0] > 0
    assert X.shape[1] > 0
    assert X.isna().sum().sum() == 0
    assert y.nunique() >= 2

    loaded_data[
        dataset_name
    ] = (
        X,
        y,
        metadata
    )

    dataset_audit.append({

        "dataset":
            dataset_name,

        "rows":
            len(X),

        "features":
            X.shape[1],

        "classes":
            y.nunique(),

        "removed_rows":
            removed,

        "missing_values":
            int(
                X.isna().sum().sum()
            ),

        "duplicate_feature_rows":
            int(
                X.duplicated().sum()
            ),

        "load_seconds":
            elapsed,

        "source":
            metadata
    })


# ================================================================
# 9. CANONICAL SPLITS
# ================================================================

print("\n[3] CANONICAL SPLITS")
print("-" * 78)

split_registry = {}

for dataset_name in DATASETS:

    X, y, _ = loaded_data[
        dataset_name
    ]

    split_registry[
        dataset_name
    ] = {}

    for seed in SEEDS:

        train_idx, test_idx, split_id = (
            make_split(
                X,
                y,
                seed
            )
        )

        split_registry[
            dataset_name
        ][
            str(seed)
        ] = {

            "train_size":
                int(
                    len(train_idx)
                ),

            "test_size":
                int(
                    len(test_idx)
                ),

            "split_hash":
                split_id
        }

        print(
            f"{dataset_name:18s} "
            f"seed={seed} "
            f"train={len(train_idx):6d} "
            f"test={len(test_idx):6d} "
            f"hash={split_id}"
        )


# ================================================================
# 10. ENGINE VALIDATION
# ================================================================

print("\n[4] 120-RUN ENGINE VALIDATION")
print("-" * 78)

results = []

run_counter = 0

for dataset_name in DATASETS:

    X, y, _ = loaded_data[
        dataset_name
    ]

    for seed in SEEDS:

        train_idx, test_idx, split_id = (
            make_split(
                X,
                y,
                seed
            )
        )

        for model_name in MODELS:

            for configuration in (
                CONFIGURATIONS.keys()
            ):

                run_counter += 1

                print(
                    f"\n[{run_counter}/{EXPECTED_RUNS}] "
                    f"{dataset_name} | "
                    f"{model_name} | "
                    f"seed={seed} | "
                    f"{configuration}"
                )

                try:

                    result = run_single(

                        dataset_name=
                            dataset_name,

                        model_name=
                            model_name,

                        seed=
                            seed,

                        configuration=
                            configuration,

                        X=X,

                        y=y,

                        train_idx=
                            train_idx,

                        test_idx=
                            test_idx,

                        split_id=
                            split_id
                    )

                    print(
                        "PASS | "
                        f"Macro-F1={result['macro_f1']:.6f} | "
                        f"BA={result['balanced_accuracy']:.6f} | "
                        f"time={result['runtime_seconds']:.2f}s"
                    )

                except Exception as exc:

                    result = {

                        "run_key":
                            f"{dataset_name}__"
                            f"{model_name}__"
                            f"seed{seed}__"
                            f"{configuration}",

                        "dataset":
                            dataset_name,

                        "model":
                            model_name,

                        "seed":
                            seed,

                        "configuration":
                            configuration,

                        "status":
                            "FAIL",

                        "error":
                            f"{type(exc).__name__}: {exc}"
                    }

                    print(
                        "FAIL |",
                        result["error"]
                    )

                results.append(
                    result
                )

                # Save checkpoint after every run.
                pd.DataFrame(
                    results
                ).to_csv(
                    BASE
                    /
                    "CTD_BATCH2_ENGINE_VALIDATION_RESULTS_CHECKPOINT.csv",
                    index=False
                )


results_df = pd.DataFrame(
    results
)


# ================================================================
# 11. VALIDATION GATES
# ================================================================

print("\n" + "=" * 78)
print("BATCH 2 VALIDATION GATES")
print("=" * 78)

# G1 — run count
g1 = (
    len(results_df)
    == EXPECTED_RUNS
)

print(
    "G1 — 120 runs recorded:",
    "PASS" if g1 else "FAIL"
)


# G2 — unique run keys
g2 = (
    "run_key" in results_df.columns
    and
    results_df["run_key"].duplicated().sum()
    == 0
)

print(
    "G2 — unique run keys:",
    "PASS" if g2 else "FAIL"
)


# G3 — all PASS
g3 = (
    "status" in results_df.columns
    and
    (
        results_df["status"]
        == "PASS"
    ).all()
)

print(
    "G3 — all runs PASS:",
    "PASS" if g3 else "FAIL"
)


# G4 — exact configuration counts
expected_per_configuration = (
    len(DATASETS)
    * len(MODELS)
    * len(SEEDS)
)

config_counts = (
    results_df[
        results_df["status"] == "PASS"
    ]
    ["configuration"]
    .value_counts()
    .to_dict()
)

g4 = (
    set(config_counts.keys())
    == set(CONFIGURATIONS.keys())
    and
    all(
        v == expected_per_configuration
        for v in config_counts.values()
    )
)

print(
    "G4 — configuration completeness:",
    "PASS" if g4 else "FAIL"
)

print(
    pd.Series(
        config_counts
    ).sort_index().to_string()
)


# G5 — model counts
expected_per_model = (
    len(DATASETS)
    * len(CONFIGURATIONS)
    * len(SEEDS)
)

model_counts = (
    results_df[
        results_df["status"] == "PASS"
    ]
    ["model"]
    .value_counts()
    .to_dict()
)

g5 = (
    set(model_counts.keys())
    == set(MODELS)
    and
    all(
        v == expected_per_model
        for v in model_counts.values()
    )
)

print(
    "G5 — model completeness:",
    "PASS" if g5 else "FAIL"
)

print(
    pd.Series(
        model_counts
    ).sort_index().to_string()
)


# G6 — seed counts
expected_per_seed = (
    len(DATASETS)
    * len(MODELS)
    * len(CONFIGURATIONS)
)

seed_counts = (
    results_df[
        results_df["status"] == "PASS"
    ]
    ["seed"]
    .value_counts()
    .to_dict()
)

g6 = (
    set(seed_counts.keys())
    == set(SEEDS)
    and
    all(
        v == expected_per_seed
        for v in seed_counts.values()
    )
)

print(
    "G6 — seed completeness:",
    "PASS" if g6 else "FAIL"
)

print(
    pd.Series(
        seed_counts
    ).sort_index().to_string()
)


# G7 — metric integrity
metric_cols = [
    "macro_f1",
    "accuracy",
    "balanced_accuracy"
]

g7 = False

if all(
    c in results_df.columns
    for c in metric_cols
):

    try:

        values = results_df[
            metric_cols
        ].to_numpy(
            dtype=float
        )

        g7 = (
            np.isfinite(
                values
            ).all()
            and
            (
                values[:, 0]
                >= 0
            ).all()
            and
            (
                values[:, 0]
                <= 1
            ).all()
        )

    except Exception:
        g7 = False

print(
    "G7 — metric integrity:",
    "PASS" if g7 else "FAIL"
)


# G8 — split reuse
g8 = True

split_counts = (
    results_df[
        results_df["status"] == "PASS"
    ]
    .groupby(
        ["dataset", "seed"]
    )["split_hash"]
    .nunique()
)

if (
    len(split_counts) !=
    len(DATASETS) * len(SEEDS)
):
    g8 = False

elif not (
    split_counts == 1
).all():
    g8 = False

print(
    "G8 — one canonical split per dataset-seed:",
    "PASS" if g8 else "FAIL"
)


# G9 — expected runs per split
g9 = True

runs_per_split = (
    results_df[
        results_df["status"] == "PASS"
    ]
    .groupby(
        ["dataset", "seed", "split_hash"]
    )
    .size()
)

expected_runs_per_split = (
    len(MODELS)
    * len(CONFIGURATIONS)
)

if (
    len(runs_per_split)
    != len(DATASETS) * len(SEEDS)
):
    g9 = False

elif not (
    runs_per_split
    == expected_runs_per_split
).all():
    g9 = False

print(
    "G9 — 20 runs per dataset-seed split:",
    "PASS" if g9 else "FAIL"
)


# G10 — deterministic repeat
#
# Repeat the first dataset/seed/model/configuration exactly once.
# This is an additional engine check, not part of the 120 rows.
# ================================================================

print("\n[5] DETERMINISTIC REPEAT CHECK")
print("-" * 78)

det_dataset = DATASETS[0]
det_seed = SEEDS[0]
det_model = "logistic_regression"
det_config = "baseline"

X_det, y_det, _ = loaded_data[
    det_dataset
]

train_det, test_det, hash_det = (
    make_split(
        X_det,
        y_det,
        det_seed
    )
)

repeat_a = run_single(
    det_dataset,
    det_model,
    det_seed,
    det_config,
    X_det,
    y_det,
    train_det,
    test_det,
    hash_det
)

repeat_b = run_single(
    det_dataset,
    det_model,
    det_seed,
    det_config,
    X_det,
    y_det,
    train_det,
    test_det,
    hash_det
)

g10 = (
    repeat_a["status"] == "PASS"
    and
    repeat_b["status"] == "PASS"
    and
    repeat_a["split_hash"]
    == repeat_b["split_hash"]
    and
    np.isclose(
        repeat_a["macro_f1"],
        repeat_b["macro_f1"],
        rtol=0,
        atol=1e-12
    )
    and
    np.isclose(
        repeat_a["accuracy"],
        repeat_b["accuracy"],
        rtol=0,
        atol=1e-12
    )
    and
    np.isclose(
        repeat_a["balanced_accuracy"],
        repeat_b["balanced_accuracy"],
        rtol=0,
        atol=1e-12
    )
)

print(
    "Repeat A Macro-F1:",
    repeat_a["macro_f1"]
)

print(
    "Repeat B Macro-F1:",
    repeat_b["macro_f1"]
)

print(
    "G10 — deterministic repeat:",
    "PASS" if g10 else "FAIL"
)


# ================================================================
# 12. RESOURCE SUMMARY
# ================================================================

print("\n[6] RESOURCE SUMMARY")
print("-" * 78)

if (
    "runtime_seconds"
    in results_df.columns
):

    resource_summary = (
        results_df[
            results_df["status"] == "PASS"
        ]
        .groupby(
            [
                "dataset",
                "model",
                "configuration"
            ]
        )["runtime_seconds"]
        .agg(
            [
                "count",
                "mean",
                "max",
                "min"
            ]
        )
        .reset_index()
    )

    print(
        resource_summary.to_string(
            index=False
        )
    )

    resource_summary.to_csv(
        BASE
        /
        "CTD_BATCH2_RESOURCE_SUMMARY.csv",
        index=False
    )


# ================================================================
# 13. FINAL REPORT
# ================================================================

dataset_audit_df = pd.DataFrame(
    dataset_audit
)

dataset_audit_df.to_csv(
    BASE
    /
    "CTD_BATCH2_DATASET_AUDIT.csv",
    index=False
)

with open(
    BASE
    /
    "CTD_BATCH2_SPLIT_REGISTRY.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        split_registry,
        f,
        indent=2
    )

results_df.to_csv(
    BASE
    /
    "CTD_BATCH2_ENGINE_VALIDATION_RESULTS.csv",
    index=False
)

all_gates = all([
    g1,
    g2,
    g3,
    g4,
    g5,
    g6,
    g7,
    g8,
    g9,
    g10
])

report = {

    "timestamp":
        datetime.now().isoformat(),

    "validation_only":
        True,

    "production_authorized":
        False,

    "datasets":
        DATASETS,

    "models":
        MODELS,

    "configurations":
        list(CONFIGURATIONS.keys()),

    "seeds":
        SEEDS,

    "expected_runs":
        EXPECTED_RUNS,

    "actual_rows":
        len(results_df),

    "gates": {

        "G1_run_count":
            bool(g1),

        "G2_unique_run_keys":
            bool(g2),

        "G3_all_pass":
            bool(g3),

        "G4_configuration_completeness":
            bool(g4),

        "G5_model_completeness":
            bool(g5),

        "G6_seed_completeness":
            bool(g6),

        "G7_metric_integrity":
            bool(g7),

        "G8_split_reuse":
            bool(g8),

        "G9_runs_per_split":
            bool(g9),

        "G10_deterministic_repeat":
            bool(g10)
    },

    "final_validation":
        (
            "PASS"
            if all_gates
            else
            "FAIL"
        ),

    "next_action":
        (
            "Do not start production until "
            "the validation output is independently reviewed."
        )
}

with open(
    BASE
    /
    "CTD_BATCH2_ENGINE_VALIDATION_REPORT.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=2
    )


# ================================================================
# 14. FINAL GATE
# ================================================================

print("\n" + "=" * 78)
print("BATCH 2 ENGINE VALIDATION — FINAL")
print("=" * 78)

for name, value in [
    ("G1 run count", g1),
    ("G2 unique run keys", g2),
    ("G3 all runs PASS", g3),
    ("G4 configuration completeness", g4),
    ("G5 model completeness", g5),
    ("G6 seed completeness", g6),
    ("G7 metric integrity", g7),
    ("G8 split reuse", g8),
    ("G9 runs per split", g9),
    ("G10 deterministic repeat", g10)
]:

    print(
        f"{name:42s}:",
        "PASS" if value else "FAIL"
    )

print("\nFINAL VALIDATION:")

print(
    "PASS — Batch 2 engine is validated."
    if all_gates
    else
    "HOLD — Batch 2 engine validation requires review."
)

print(
    "\nIMPORTANT:"
)

print(
    "Production authorization: FALSE"
)

print(
    "NO 900-run production campaign was started."
)

print(
    "\nOutput directory:",
    BASE
)

print(
    "Results:",
    BASE /
    "CTD_BATCH2_ENGINE_VALIDATION_RESULTS.csv"
)

print(
    "Dataset audit:",
    BASE /
    "CTD_BATCH2_DATASET_AUDIT.csv"
)

print(
    "Split registry:",
    BASE /
    "CTD_BATCH2_SPLIT_REGISTRY.json"
)

print(
    "Report:",
    BASE /
    "CTD_BATCH2_ENGINE_VALIDATION_REPORT.json"
)

print("=" * 78)
