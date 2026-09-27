# ================================================================
# CTD IJDSA REVISION — BATCH 2B SPAMBASE PRODUCTION v1
# ================================================================
# FROZEN SCIENTIFIC PROTOCOL — DO NOT MODIFY AFTER STARTING
# Dataset: Spambase
# 300 production fits = 15 seeds × 4 models × 5 configurations
# Primary metric: Macro-F1
# Production authorization is for Batch 2B only.
# ================================================================

import os
import json
import time
import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score

warnings.filterwarnings("ignore")

# ================================================================
# 0. FROZEN DESIGN
# ================================================================

DATASET = "spambase"
OPENML_ID = 44

SEEDS = [42, 123, 2026, 7, 19, 31, 53, 71, 89, 107, 149, 173, 211, 257, 311]

MODELS = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "linear_svm",
]

CONFIGURATIONS = {
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

EXPECTED_RUNS = 300
TEST_SIZE = 0.20

OUTDIR = Path("/content/CTD_BATCH2B_SPAMBASE_PRODUCTION")
OUTDIR.mkdir(parents=True, exist_ok=True)

RESULT_FILE = OUTDIR / "CTD_BATCH2B_SPAMBASE_RESULTS_LEDGER.csv"
STATUS_FILE = OUTDIR / "CTD_BATCH2B_SPAMBASE_RUN_STATUS.csv"
CHECKPOINT_FILE = OUTDIR / "CTD_BATCH2B_SPAMBASE_CHECKPOINT.json"
MANIFEST_FILE = OUTDIR / "CTD_BATCH2B_SPAMBASE_PRODUCTION_MANIFEST.json"
REPORT_FILE = OUTDIR / "CTD_BATCH2B_SPAMBASE_FINAL_VALIDATION_REPORT.json"

print("=" * 72)
print("CTD IJDSA REVISION — BATCH 2B SPAMBASE PRODUCTION v1")
print("=" * 72)
print(f"Dataset        : {DATASET}")
print(f"OpenML ID      : {OPENML_ID}")
print(f"Seeds          : {SEEDS}")
print(f"Models         : {MODELS}")
print(f"Configurations : {list(CONFIGURATIONS.keys())}")
print(f"Expected fits  : {EXPECTED_RUNS}")
print("Primary metric : Macro-F1")
print("Production     : TRUE — Batch 2B only")

# ================================================================
# 1. ENVIRONMENT
# ================================================================

import sklearn

print("\n[1] ENVIRONMENT")
print("-" * 72)
print("Python       :", __import__("sys").version.split()[0])
print("NumPy        :", np.__version__)
print("Pandas       :", pd.__version__)
print("Scikit-learn :", sklearn.__version__)

# ================================================================
# 2. HELPERS
# ================================================================

def sha256_array(arr):
    a = np.asarray(arr)
    return hashlib.sha256(
        np.ascontiguousarray(a).tobytes()
    ).hexdigest()

def sha256_dataframe(df):
    payload = pd.util.hash_pandas_object(
        df, index=True
    ).values.tobytes()
    return hashlib.sha256(payload).hexdigest()

def canonical_split_hash(train_idx, test_idx):
    payload = (
        np.asarray(train_idx, dtype=np.int64).tobytes()
        + np.asarray(test_idx, dtype=np.int64).tobytes()
    )
    return hashlib.sha256(payload).hexdigest()

def deterministic_oversample(X_train, y_train, seed):
    rng = np.random.RandomState(seed)

    y_arr = np.asarray(y_train)
    classes, counts = np.unique(y_arr, return_counts=True)
    max_count = int(counts.max())

    selected_parts = []
    selected_labels = []

    for cls, count in zip(classes, counts):
        idx = np.where(y_arr == cls)[0]

        if count < max_count:
            extra = rng.choice(
                idx,
                size=max_count - count,
                replace=True
            )
            use_idx = np.concatenate([idx, extra])
        else:
            use_idx = idx.copy()

        selected_parts.append(X_train[use_idx])
        selected_labels.append(y_arr[use_idx])

    X_out = np.vstack(selected_parts)
    y_out = np.concatenate(selected_labels)

    # Deterministic shuffle after class balancing
    order = rng.permutation(len(y_out))

    return X_out[order], y_out[order]

def make_model(model_name, seed):
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

    raise ValueError(model_name)

def load_spambase():
    bunch = fetch_openml(
        data_id=OPENML_ID,
        as_frame=True
    )

    X = bunch.data.copy()
    y = bunch.target.copy()

    # Ensure numeric predictors
    X = X.apply(pd.to_numeric, errors="raise")

    # Normalize target to deterministic binary integer labels
    y_str = y.astype(str).str.strip().str.lower()

    unique_y = sorted(y_str.unique().tolist())

    if len(unique_y) != 2:
        raise AssertionError(
            f"Expected 2 target classes, found {unique_y}"
        )

    # OpenML Spambase uses spam/non-spam target labels.
    # Map deterministically by sorted label order.
    mapping = {label: i for i, label in enumerate(unique_y)}
    y_num = y_str.map(mapping).astype(int)

    return X, y_num, unique_y, mapping

# ================================================================
# 3. DATASET AUDIT / IDENTITY
# ================================================================

print("\n[2] LOADING SPAMBASE DATASET")
print("-" * 72)

X, y, target_labels, target_mapping = load_spambase()

print("Shape              :", X.shape)
print("Classes            :", y.nunique())
print("Missing values     :", int(X.isna().sum().sum()))
print(
    "Duplicate features :",
    int(X.duplicated().sum())
)

assert X.shape == (4601, 57), X.shape
assert y.nunique() == 2
assert X.isna().sum().sum() == 0

feature_hash = sha256_dataframe(X)
target_hash = sha256_array(y.to_numpy())
combined_hash = hashlib.sha256(
    (
        feature_hash + target_hash
    ).encode("utf-8")
).hexdigest()

print("\nDATASET FINGERPRINT")
print("-" * 72)
print("Feature SHA-256 :", feature_hash)
print("Target SHA-256  :", target_hash)
print("Combined SHA-256:", combined_hash)
print("Target labels   :", target_labels)
print("Target mapping  :", target_mapping)

# ================================================================
# 4. CANONICAL SPLITS
# ================================================================

print("\n[3] CANONICAL SPLITS")
print("-" * 72)

split_registry = {}

for seed in SEEDS:
    all_idx = np.arange(len(X))

    train_idx, test_idx = train_test_split(
        all_idx,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=y
    )

    train_idx = np.asarray(train_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)

    split_hash = canonical_split_hash(
        train_idx,
        test_idx
    )

    split_registry[str(seed)] = {
        "train_idx": train_idx.tolist(),
        "test_idx": test_idx.tolist(),
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "split_hash": split_hash,
    }

    print(
        f"seed={seed:<4} | "
        f"train={len(train_idx):5d} | "
        f"test={len(test_idx):5d} | "
        f"hash={split_hash}"
    )

# ================================================================
# 5. MANIFEST
# ================================================================

manifest_rows = []

for seed in SEEDS:
    split_hash = split_registry[str(seed)]["split_hash"]

    for model_name in MODELS:
        for config_name in CONFIGURATIONS.keys():

            run_key = (
                f"{DATASET}__{model_name}__"
                f"seed{seed}__{config_name}"
            )

            manifest_rows.append({
                "run_key": run_key,
                "dataset": DATASET,
                "openml_id": OPENML_ID,
                "model": model_name,
                "seed": seed,
                "configuration": config_name,
                "scaling": CONFIGURATIONS[config_name]["scaling"],
                "sampling": CONFIGURATIONS[config_name]["sampling"],
                "representation": CONFIGURATIONS[config_name]["representation"],
                "training_fraction": CONFIGURATIONS[config_name]["training_fraction"],
                "split_hash": split_hash,
            })

manifest_df = pd.DataFrame(manifest_rows)

assert len(manifest_df) == EXPECTED_RUNS
assert manifest_df["run_key"].nunique() == EXPECTED_RUNS

print("\n[4] PRE-PRODUCTION ACCOUNTING")
print("-" * 72)
print("Manifest rows :", len(manifest_df))
print("Unique keys   :", manifest_df["run_key"].nunique())
print("5 configurations × 60 :", (len(SEEDS) * len(MODELS)))
print("4 models × 75         :", (len(SEEDS) * len(CONFIGURATIONS)))
print("15 seeds × 20         :", (len(MODELS) * len(CONFIGURATIONS)))

assert (
    manifest_df.groupby("configuration").size()
    == 60
).all()

assert (
    manifest_df.groupby("model").size()
    == 75
).all()

assert (
    manifest_df.groupby("seed").size()
    == 20
).all()

print("Pre-production gate: PASS")

# ================================================================
# 6. SAVE MANIFEST
# ================================================================

manifest_payload = {
    "batch": "2B",
    "dataset": DATASET,
    "openml_id": OPENML_ID,
    "dataset_shape": list(X.shape),
    "n_classes": int(y.nunique()),
    "feature_sha256": feature_hash,
    "target_sha256": target_hash,
    "combined_sha256": combined_hash,
    "target_labels": target_labels,
    "target_mapping": target_mapping,
    "seeds": SEEDS,
    "models": MODELS,
    "configurations": CONFIGURATIONS,
    "expected_runs": EXPECTED_RUNS,
    "primary_metric": "macro_f1",
    "test_size": TEST_SIZE,
    "split_registry": {
        str(seed): {
            "train_size": split_registry[str(seed)]["train_size"],
            "test_size": split_registry[str(seed)]["test_size"],
            "split_hash": split_registry[str(seed)]["split_hash"],
        }
        for seed in SEEDS
    },
    "production_authorized": True,
}

with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
    json.dump(manifest_payload, f, indent=2)

# ================================================================
# 7. RESUME SUPPORT
# ================================================================

if RESULT_FILE.exists():
    results_df = pd.read_csv(RESULT_FILE)
else:
    results_df = pd.DataFrame()

if STATUS_FILE.exists():
    status_df = pd.read_csv(STATUS_FILE)
else:
    status_df = pd.DataFrame()

existing_pass_keys = set()

if not results_df.empty and "status" in results_df.columns:
    existing_pass_keys = set(
        results_df.loc[
            results_df["status"].astype(str).eq("PASS"),
            "run_key"
        ].astype(str)
    )

print("\nExisting PASS runs:", len(existing_pass_keys))

# ================================================================
# 8. SINGLE EXPERIMENT
# ================================================================

def run_single_experiment(
    dataset_name,
    model_name,
    seed,
    configuration,
    X,
    y,
    train_idx,
    test_idx,
    split_hash
):

    cfg = CONFIGURATIONS[configuration]

    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()

    y_train = y.iloc[train_idx].to_numpy()
    y_test = y.iloc[test_idx].to_numpy()

    # Training fraction counterfactual
    if cfg["training_fraction"] < 1.0:
        rng = np.random.RandomState(seed)

        keep_n = int(
            round(
                len(X_train) *
                cfg["training_fraction"]
            )
        )

        # Stratified deterministic subset
        selected = []

        for cls in np.unique(y_train):
            cls_idx = np.where(y_train == cls)[0]
            n_cls = max(
                1,
                int(
                    round(
                        len(cls_idx) *
                        cfg["training_fraction"]
                    )
                )
            )
            selected.extend(
                rng.choice(
                    cls_idx,
                    size=min(n_cls, len(cls_idx)),
                    replace=False
                ).tolist()
            )

        selected = np.asarray(
            sorted(set(selected)),
            dtype=int
        )

        X_train = X_train.iloc[selected].copy()
        y_train = y_train[selected]

    # Scaling
    if cfg["scaling"]:
        scaler = StandardScaler()
        X_train_proc = scaler.fit_transform(X_train)
        X_test_proc = scaler.transform(X_test)
    else:
        X_train_proc = X_train.to_numpy()
        X_test_proc = X_test.to_numpy()

    # Sampling — training only
    if cfg["sampling"] == "minority_oversampling":
        X_train_proc, y_train = deterministic_oversample(
            X_train_proc,
            y_train,
            seed
        )

    # Representation — training only
    if cfg["representation"] == "pca95":
        pca = PCA(
            n_components=0.95,
            svd_solver="full"
        )
        X_train_proc = pca.fit_transform(X_train_proc)
        X_test_proc = pca.transform(X_test_proc)

    model = make_model(model_name, seed)

    start = time.time()

    model.fit(
        X_train_proc,
        y_train
    )

    elapsed = time.time() - start

    pred = model.predict(X_test_proc)

    macro_f1 = f1_score(
        y_test,
        pred,
        average="macro"
    )

    accuracy = accuracy_score(
        y_test,
        pred
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        pred
    )

    metrics = {
        "macro_f1": float(macro_f1),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
    }

    for k, v in metrics.items():
        if not np.isfinite(v):
            raise ValueError(
                f"Invalid metric {k}={v}"
            )

    return {
        "dataset": dataset_name,
        "openml_id": OPENML_ID,
        "model": model_name,
        "seed": seed,
        "configuration": configuration,
        "scaling": cfg["scaling"],
        "sampling": cfg["sampling"],
        "representation": cfg["representation"],
        "training_fraction": cfg["training_fraction"],
        "split_hash": split_hash,
        "macro_f1": metrics["macro_f1"],
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "runtime_seconds": float(elapsed),
        "status": "PASS",
    }

# ================================================================
# 9. PRODUCTION EXECUTION
# ================================================================

print("\n" + "=" * 72)
print("BATCH 2B — SPAMBASE PRODUCTION EXECUTION")
print("=" * 72)

run_counter = 0

for _, row in manifest_df.iterrows():

    run_counter += 1

    run_key = row["run_key"]

    if run_key in existing_pass_keys:
        print(
            f"[SKIP {run_counter:03d}/{EXPECTED_RUNS}] "
            f"{run_key} already PASS"
        )
        continue

    seed = int(row["seed"])
    model_name = row["model"]
    config_name = row["configuration"]

    train_idx = np.asarray(
        split_registry[str(seed)]["train_idx"],
        dtype=np.int64
    )

    test_idx = np.asarray(
        split_registry[str(seed)]["test_idx"],
        dtype=np.int64
    )

    split_hash = split_registry[str(seed)]["split_hash"]

    print(
        f"\n[{run_counter:03d}/{EXPECTED_RUNS}] "
        f"{model_name} | seed={seed} | {config_name}"
    )

    try:

        result = run_single_experiment(
            DATASET,
            model_name,
            seed,
            config_name,
            X,
            y,
            train_idx,
            test_idx,
            split_hash
        )

        result["run_key"] = run_key

        # Append result
        if RESULT_FILE.exists():
            old = pd.read_csv(RESULT_FILE)
        else:
            old = pd.DataFrame()

        new = pd.DataFrame([result])
        combined = pd.concat(
            [old, new],
            ignore_index=True
        )

        combined.to_csv(
            RESULT_FILE,
            index=False
        )

        existing_pass_keys.add(run_key)

        print(
            f"PASS | Macro-F1={result['macro_f1']:.6f} | "
            f"BA={result['balanced_accuracy']:.6f} | "
            f"time={result['runtime_seconds']:.3f}s"
        )

    except Exception as exc:

        fail_result = {
            "run_key": run_key,
            "dataset": DATASET,
            "openml_id": OPENML_ID,
            "model": model_name,
            "seed": seed,
            "configuration": config_name,
            "scaling": row["scaling"],
            "sampling": row["sampling"],
            "representation": row["representation"],
            "training_fraction": row["training_fraction"],
            "split_hash": split_hash,
            "macro_f1": np.nan,
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "runtime_seconds": np.nan,
            "status": "FAIL",
            "error": repr(exc),
        }

        if RESULT_FILE.exists():
            old = pd.read_csv(RESULT_FILE)
        else:
            old = pd.DataFrame()

        combined = pd.concat(
            [old, pd.DataFrame([fail_result])],
            ignore_index=True
        )

        combined.to_csv(
            RESULT_FILE,
            index=False
        )

        print(
            "FAIL |",
            repr(exc)
        )

    # checkpoint after every run
    checkpoint = {
        "batch": "2B",
        "dataset": DATASET,
        "expected_runs": EXPECTED_RUNS,
        "pass_runs": len(existing_pass_keys),
        "last_run_counter": run_counter,
        "timestamp": pd.Timestamp.now().isoformat(),
    }

    with open(
        CHECKPOINT_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            checkpoint,
            f,
            indent=2
        )

    # Stop immediately if a run fails.
    # We audit before any continuation.
    if RESULT_FILE.exists():
        latest = pd.read_csv(RESULT_FILE)
        latest_row = latest.iloc[-1]

        if str(latest_row["status"]) != "PASS":
            print("\nSTOPPING AFTER FAILURE.")
            print("Audit the failure before resuming.")
            break

# ================================================================
# 10. FINAL VALIDATION
# ================================================================

print("\n" + "=" * 72)
print("BATCH 2B — FINAL VALIDATION")
print("=" * 72)

assert RESULT_FILE.exists(), "Missing result ledger"

final_df = pd.read_csv(RESULT_FILE)

# G1
g1 = len(final_df) == EXPECTED_RUNS
print(f"G1 — 300 rows: {'PASS' if g1 else 'FAIL'}")

# G2
g2 = (
    final_df["run_key"].nunique() == EXPECTED_RUNS
)
print(f"G2 — unique keys: {'PASS' if g2 else 'FAIL'}")

# G3
g3 = (
    len(final_df) == EXPECTED_RUNS
    and final_df["status"].astype(str).eq("PASS").all()
)
print(f"G3 — all runs PASS: {'PASS' if g3 else 'FAIL'}")

# G4
config_counts = (
    final_df["configuration"]
    .value_counts()
    .to_dict()
)

g4 = (
    set(config_counts.keys())
    == set(CONFIGURATIONS.keys())
    and all(v == 60 for v in config_counts.values())
)

print(f"G4 — 60/configuration: {'PASS' if g4 else 'FAIL'}")
print(config_counts)

# G5
model_counts = (
    final_df["model"]
    .value_counts()
    .to_dict()
)

g5 = (
    set(model_counts.keys()) == set(MODELS)
    and all(v == 75 for v in model_counts.values())
)

print(f"G5 — 75/model: {'PASS' if g5 else 'FAIL'}")
print(model_counts)

# G6
seed_counts = (
    final_df["seed"]
    .value_counts()
    .to_dict()
)

g6 = (
    set(map(int, seed_counts.keys()))
    == set(SEEDS)
    and all(v == 20 for v in seed_counts.values())
)

print(f"G6 — 20/seed: {'PASS' if g6 else 'FAIL'}")
print(seed_counts)

# G7
metric_cols = [
    "macro_f1",
    "accuracy",
    "balanced_accuracy",
    "runtime_seconds",
]

g7 = (
    final_df[metric_cols]
    .notna()
    .all()
    .all()
    and np.isfinite(
        final_df[metric_cols].to_numpy(
            dtype=float
        )
    ).all()
    and (
        final_df["runtime_seconds"] > 0
    ).all()
)

print(f"G7 — metric integrity: {'PASS' if g7 else 'FAIL'}")

# G8
split_counts = (
    final_df.groupby("seed")["split_hash"]
    .nunique()
)

g8 = (
    len(split_counts) == len(SEEDS)
    and (split_counts == 1).all()
)

print(f"G8 — one split/seed: {'PASS' if g8 else 'FAIL'}")

# G9
split_run_counts = (
    final_df.groupby(
        ["seed", "split_hash"]
    ).size()
)

g9 = (
    len(split_run_counts) == len(SEEDS)
    and (split_run_counts == 20).all()
)

print(f"G9 — 20 runs/split: {'PASS' if g9 else 'FAIL'}")

# G10 — dataset isolation and identity
g10 = (
    final_df["dataset"].astype(str).eq(DATASET).all()
    and final_df["openml_id"].astype(int).eq(OPENML_ID).all()
)

print(f"G10 — Spambase isolation: {'PASS' if g10 else 'FAIL'}")

# G11 — exact run-key coverage
expected_keys = set(manifest_df["run_key"])
actual_keys = set(final_df["run_key"])

g11 = expected_keys == actual_keys

print(f"G11 — exact run keys: {'PASS' if g11 else 'FAIL'}")

# ================================================================
# 11. FINAL REPORT
# ================================================================

all_pass = all([
    g1, g2, g3, g4, g5, g6,
    g7, g8, g9, g10, g11
])

report = {
    "batch": "2B",
    "dataset": DATASET,
    "openml_id": OPENML_ID,
    "expected_runs": EXPECTED_RUNS,
    "recorded_runs": int(len(final_df)),
    "pass_runs": int(
        final_df["status"].astype(str).eq("PASS").sum()
    ),
    "fail_runs": int(
        final_df["status"].astype(str).eq("FAIL").sum()
    ),
    "gates": {
        "G1_rows": bool(g1),
        "G2_unique_keys": bool(g2),
        "G3_all_pass": bool(g3),
        "G4_configuration_counts": bool(g4),
        "G5_model_counts": bool(g5),
        "G6_seed_counts": bool(g6),
        "G7_metric_integrity": bool(g7),
        "G8_one_split_per_seed": bool(g8),
        "G9_twenty_runs_per_split": bool(g9),
        "G10_dataset_isolation": bool(g10),
        "G11_exact_run_keys": bool(g11),
    },
    "final_validation": bool(all_pass),
    "production_complete": bool(all_pass),
    "feature_sha256": feature_hash,
    "target_sha256": target_hash,
    "combined_sha256": combined_hash,
    "split_registry": {
        str(seed): {
            "train_size": split_registry[str(seed)]["train_size"],
            "test_size": split_registry[str(seed)]["test_size"],
            "split_hash": split_registry[str(seed)]["split_hash"],
        }
        for seed in SEEDS
    },
}

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        report,
        f,
        indent=2
    )

print("\nPASS runs:", report["pass_runs"])
print("FAIL runs:", report["fail_runs"])

print("\nFINAL VALIDATION:")
if all_pass:
    print("PASS — Batch 2B Spambase production complete.")
else:
    print("HOLD — Batch 2B requires audit before any continuation.")

print("\nResults :", RESULT_FILE)
print("Status  :", STATUS_FILE)
print("Checkpoint:", CHECKPOINT_FILE)
print("Manifest:", MANIFEST_FILE)
print("Final report:", REPORT_FILE)

print("\n" + "=" * 72)
print("IMPORTANT")
print("=" * 72)
print(
    "Do NOT start Batch 2C Phoneme until this Batch 2B "
    "output has been reviewed and all gates are PASS."
)
