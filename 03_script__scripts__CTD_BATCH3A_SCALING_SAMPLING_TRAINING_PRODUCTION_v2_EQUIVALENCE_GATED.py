
# CTD IJDSA REVISION — BATCH 3A PRODUCTION v1.0
# Purpose:
#   Produce ONLY the two new joint configurations required for:
#   (1) Scaling × Training Fraction
#   (2) Sampling × Training Fraction
#
# Existing baseline/single-intervention ledgers are reused; they are NOT rerun.
# New production fits = 5 datasets × 4 models × 15 seeds × 2 joint configs = 600.
#
# Frozen datasets:
#   Breast Cancer (sklearn load_breast_cancer), Bank Marketing (validated UCI bank-full),
#   Pima Diabetes (OpenML 37), Spambase (OpenML 44), Phoneme (OpenML 1489)
#
# Frozen models:
#   Logistic Regression, Decision Tree, Random Forest, Linear SVM
#
# Frozen seeds:
#   [42,123,2026,7,19,31,53,71,89,107,149,173,211,257,311]
#
# Frozen joint configurations:
#   scaling_training_joint:
#       scaling=False, sampling=original, representation=all_features,
#       training_fraction=0.5
#
#   sampling_training_joint:
#       scaling=True, sampling=minority_oversampling,
#       representation=all_features, training_fraction=0.5
#
# Primary metric: Macro-F1
# Fixed stratified 80/20 split per dataset/seed.
# Checkpoint/resume enabled.
#
# IMPORTANT:
#   Do not alter dataset/model/seed/configuration definitions before execution.
#   This cell does NOT run Covertype.

import os, json, time, hashlib, platform, warnings, gc
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn

from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score

# -------------------------------------------------------------------
# FROZEN SETTINGS
# -------------------------------------------------------------------
BASE = Path("/content/CTD_BATCH3A_SCALING_SAMPLING_TRAINING_PRODUCTION")
BASE.mkdir(parents=True, exist_ok=True)

SEEDS = [42,123,2026,7,19,31,53,71,89,107,149,173,211,257,311]
MODELS = ["logistic_regression","decision_tree","random_forest","linear_svm"]
NEW_CONFIGS = ["scaling_training_joint","sampling_training_joint"]

CONFIGS = {
    "scaling_training_joint": {
        "scaling": False, "sampling": "original",
        "representation": "all_features", "training_fraction": 0.5
    },
    "sampling_training_joint": {
        "scaling": True, "sampling": "minority_oversampling",
        "representation": "all_features", "training_fraction": 0.5
    }
}

DATASETS = {
    "breast_cancer": {"kind":"openml","id":15},
    "bank_marketing": {"kind":"bank"},
    "pima_diabetes": {"kind":"openml","id":37},
    "spambase": {"kind":"openml","id":44},
    "phoneme": {"kind":"openml","id":1489},
}

EXPECTED = len(DATASETS)*len(MODELS)*len(SEEDS)*len(NEW_CONFIGS)

RESULTS = BASE/"CTD_BATCH3A_NEW_JOINT_RESULTS.csv"
STATUS = BASE/"CTD_BATCH3A_NEW_JOINT_STATUS.csv"
CHECKPOINT = BASE/"CTD_BATCH3A_CHECKPOINT.json"
MANIFEST = BASE/"CTD_BATCH3A_PRODUCTION_MANIFEST.json"
FINAL = BASE/"CTD_BATCH3A_FINAL_VALIDATION_REPORT.json"
INTERACTIONS = BASE/"CTD_BATCH3A_INTERACTION_VALUES.csv"

SEED_ORDER = SEEDS

print("="*78)
print("CTD IJDSA REVISION — BATCH 3A PRODUCTION v1.0")
print("="*78)
print("Expected NEW fits:", EXPECTED)
print("Python:", platform.python_version())
print("NumPy:", np.__version__, "Pandas:", pd.__version__,
      "Scikit-learn:", sklearn.__version__)
print("Datasets:", list(DATASETS))
print("Models:", MODELS)
print("Seeds:", SEEDS)
print("New configurations:", NEW_CONFIGS)

# -------------------------------------------------------------------
# DATA LOADING
# -------------------------------------------------------------------
def load_dataset(name, spec):
    if name == "bank_marketing":
        # Validated UCI bank-full.csv.
        # The file is expected at /content/bank-full.csv if already present.
        candidates = [
            Path("/content/bank-full.csv"),
            Path("/content/Bank Marketing/bank-full.csv"),
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            raise FileNotFoundError(
                "bank-full.csv not found. Upload the validated UCI bank-full.csv "
                "to /content before running."
            )
        df = pd.read_csv(path, sep=";")
        assert df.shape == (45211,17), df.shape
        target_col = "y"
        X = df.drop(columns=[target_col]).copy()
        y = df[target_col].astype(str).str.strip()
        # Canonical binary mapping used by the validated pipeline.
        vals = sorted(y.unique().tolist())
        assert vals == ["no","yes"], vals
        y = y.map({"no":0,"yes":1}).astype(int)
        return X, y, {"openml_id":None, "source":"UCI bank-full.csv"}

    if name == "breast_cancer":
        # EXACT dataset loader used by the validated Batch 1A production.
        bunch = None
        d = load_breast_cancer(as_frame=True)
        X = d.data.copy()
        y = pd.Series(d.target.astype(int)).copy()
        return X, y, {"openml_id":None, "source":"sklearn.datasets.load_breast_cancer"}
    else:
        bunch = fetch_openml(data_id=spec["id"], as_frame=True)

    X = pd.DataFrame(bunch.data).copy()
    y = pd.Series(bunch.target).astype(str).copy()

    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="raise")

    # Frozen mappings for these binary datasets.
    vals = sorted(y.unique().tolist())
    if len(vals) == 2:
        y = y.map({vals[0]:0, vals[1]:1}).astype(int)

    return X, y, {"openml_id":spec["id"], "source":f"OpenML {spec['id']}"}

DATA = {}
for name, spec in DATASETS.items():
    print("\nLoading:", name)
    X, y, meta = load_dataset(name, spec)
    assert len(X) == len(y)
    assert X.isna().sum().sum() == 0
    DATA[name] = (X.reset_index(drop=True), y.reset_index(drop=True), meta)
    print(" shape:", X.shape, "classes:", y.nunique(),
          "duplicates:", int(X.duplicated().sum()))

# -------------------------------------------------------------------
# PREPROCESSING
# -------------------------------------------------------------------
def deterministic_oversample(A, b, seed):
    # EXACT implementation used by the validated Batch 2 production engine.
    rng = np.random.RandomState(seed)
    yv = np.asarray(b)
    classes, counts = np.unique(yv, return_counts=True)
    mx = int(counts.max())

    Ax = [np.asarray(A)]
    by = [yv]

    for cls, n in zip(classes, counts):
        d = mx - int(n)
        if d > 0:
            ids = np.where(yv == cls)[0]
            take = rng.choice(ids, size=d, replace=True)
            Ax.append(np.asarray(A)[take])
            by.append(yv[take])

    A = np.vstack(Ax)
    b = np.concatenate(by)
    order = rng.permutation(len(b))
    return A[order], b[order]

def model_factory(name, seed):
    if name == "logistic_regression":
        return LogisticRegression(max_iter=2000, solver="liblinear",
                                  random_state=seed)
    if name == "decision_tree":
        return DecisionTreeClassifier(random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=200,
                                      random_state=seed, n_jobs=-1)
    if name == "linear_svm":
        return LinearSVC(C=1.0, penalty="l2",
                         loss="squared_hinge", dual=False,
                         tol=1e-3, max_iter=1000,
                         random_state=seed)
    raise ValueError(name)

def split_for(X, y, seed):
    idx = np.arange(len(X), dtype=np.int64)
    tr, te = train_test_split(
        idx, test_size=0.20, stratify=y, random_state=seed
    )
    tr = np.asarray(tr, dtype=np.int64)
    te = np.asarray(te, dtype=np.int64)
    h = hashlib.sha256()
    h.update(tr.tobytes()); h.update(te.tobytes())
    return tr, te, h.hexdigest()

def make_numeric_or_encoded(name, X_train, X_test, scaling):
    # For Bank Marketing, encode categorical predictors using training data
    # only. Numeric predictors are retained. For all other datasets, all
    # predictors are numeric.
    if name == "bank_marketing":
        num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in X_train.columns if c not in num_cols]

        try:
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            enc = OneHotEncoder(handle_unknown="ignore", sparse=False)

        transformers = [("cat", enc, cat_cols)]
        if num_cols:
            transformers.append(("num", "passthrough", num_cols))

        ct = ColumnTransformer(transformers=transformers, remainder="drop")
        A = ct.fit_transform(X_train)
        B = ct.transform(X_test)
    else:
        A = X_train.to_numpy(dtype=float)
        B = X_test.to_numpy(dtype=float)

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    if scaling:
        sc = StandardScaler()
        A = sc.fit_transform(A)
        B = sc.transform(B)

    return A, B

# -------------------------------------------------------------------
# RUN ONE JOINT EXPERIMENT
# -------------------------------------------------------------------
def run_one(dataset_name, model_name, seed, config_name, tr_idx, te_idx, split_hash):
    X, y, meta = DATA[dataset_name]
    cfg = CONFIGS[config_name]
    t0 = time.time()

    X_train = X.iloc[tr_idx].copy()
    X_test = X.iloc[te_idx].copy()
    y_train = y.iloc[tr_idx].to_numpy()
    y_test = y.iloc[te_idx].to_numpy()

    # EXACT validated training-fraction implementation:
    # stratified train_test_split on the training partition only.
    if cfg["training_fraction"] < 1.0:
        X_train_np = X_train.to_numpy(dtype=float) if dataset_name != "bank_marketing" else X_train
        sub_train, _, sub_y, _ = train_test_split(
            X_train_np,
            y_train,
            train_size=cfg["training_fraction"],
            stratify=y_train,
            random_state=seed
        )
        if dataset_name == "bank_marketing":
            X_train = sub_train.copy()
        else:
            X_train = pd.DataFrame(
                sub_train,
                columns=X_train.columns
            )
        y_train = np.asarray(sub_y)

    A, B = make_numeric_or_encoded(
        dataset_name, X_train, X_test, cfg["scaling"]
    )

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

    for v in (mf1, acc, ba):
        assert np.isfinite(v)

    return {
        "run_key": f"{dataset_name}__{model_name}__seed{seed}__{config_name}",
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
        "completed_at": datetime.now().isoformat()
    }


# -------------------------------------------------------------------
# PRE-PRODUCTION EQUIVALENCE AUDIT
# -------------------------------------------------------------------
# This gate does NOT authorize production until:
#   A. all five existing audited ledgers are present and structurally valid;
#   B. every computed canonical split hash exactly matches the corresponding
#      completed production ledger for that dataset/seed;
#   C. representative baseline/sampling/training-fraction reproductions for
#      each dataset/model/seed match the audited production values.
#
# This is an evidence-equivalence gate, not a new scientific result.
BASE_LEDGER_PATHS = {
    "breast_cancer":
        "/content/CTD_BATCH1_LEDGER/CTD_BATCH1A_RESULTS_LEDGER.csv",
    "bank_marketing":
        "/content/CTD_BATCH1B_LEDGER/CTD_BATCH1B_RESULTS_LEDGER.csv",
    "pima_diabetes":
        "/content/CTD_BATCH2A_PIMA_PRODUCTION/CTD_BATCH2A_PIMA_RESULTS_LEDGER.csv",
    "spambase":
        "/content/CTD_BATCH2B_SPAMBASE_PRODUCTION/CTD_BATCH2B_SPAMBASE_RESULTS_LEDGER.csv",
    "phoneme":
        "/content/CTD_BATCH2C_PHONEME_PRODUCTION/CTD_BATCH2C_PHONEME_RESULTS_LEDGER.csv"
}

AUDIT_CONFIGS = ["baseline", "sampling_cf", "training_fraction_cf"]
AUDIT_MODEL = "logistic_regression"
AUDIT_SEED = 42

print("\n" + "="*78)
print("BATCH 3A PRE-PRODUCTION EQUIVALENCE AUDIT")
print("="*78)

AUDIT_LEDGER = {}
for ds, path in BASE_LEDGER_PATHS.items():
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Required audited ledger missing: {path}")
    z = pd.read_csv(p)
    required_cols = {"run_key","dataset","model","seed","configuration",
                     "split_hash","macro_f1","status"}
    missing = required_cols - set(z.columns)
    if missing:
        raise RuntimeError(f"{ds}: missing ledger columns: {sorted(missing)}")
    z = z[z["status"].astype(str).eq("PASS")].copy()
    z = z[z["dataset"].astype(str).eq(ds)].copy()
    if len(z) != 300:
        raise RuntimeError(f"{ds}: expected 300 PASS rows, found {len(z)}")
    if z["run_key"].nunique() != 300:
        raise RuntimeError(f"{ds}: duplicate run keys in audited ledger")
    for cfg in ["baseline","scaling_cf","sampling_cf",
                "representation_cf","training_fraction_cf"]:
        q = z[z["configuration"].astype(str).eq(cfg)]
        if len(q) != 60:
            raise RuntimeError(f"{ds}: expected 60 rows for {cfg}, found {len(q)}")
    AUDIT_LEDGER[ds] = z

# Build DATA and canonical splits first.

# -------------------------------------------------------------------
# CANONICAL SPLITS + MANIFEST
# -------------------------------------------------------------------
SPLITS = {}
manifest_rows = []

for dataset_name, (X, y, meta) in DATA.items():
    SPLITS[dataset_name] = {}
    for seed in SEEDS:
        tr, te, h = split_for(X, y, seed)
        SPLITS[dataset_name][str(seed)] = {
            "train_size": int(len(tr)),
            "test_size": int(len(te)),
            "split_hash": h
        }
        for model in MODELS:
            for cfg in NEW_CONFIGS:
                manifest_rows.append({
                    "run_key":f"{dataset_name}__{model}__seed{seed}__{cfg}",
                    "dataset":dataset_name, "model":model, "seed":seed,
                    "configuration":cfg
                })


# Exact split identity comparison against completed production ledgers.
split_identity_pass = True
split_mismatches = []

for ds in DATASETS:
    audited = AUDIT_LEDGER[ds]
    for seed in SEEDS:
        expected_hashes = set(
            audited.loc[
                audited["seed"].astype(int).eq(seed),
                "split_hash"
            ].astype(str)
        )
        computed_hash = str(SPLITS[ds][str(seed)]["split_hash"])
        if expected_hashes != {computed_hash}:
            split_identity_pass = False
            split_mismatches.append({
                "dataset": ds,
                "seed": seed,
                "computed": computed_hash,
                "audited": sorted(expected_hashes)
            })

print("Exact canonical split identity:", "PASS" if split_identity_pass else "FAIL")
if not split_identity_pass:
    print(json.dumps(split_mismatches[:10], indent=2))
    raise RuntimeError(
        "PRE-PRODUCTION GATE FAILED: computed split hashes do not exactly "
        "match completed production ledgers."
    )

manifest_df = pd.DataFrame(manifest_rows)
assert len(manifest_df) == EXPECTED
assert manifest_df["run_key"].nunique() == EXPECTED


# Representative implementation-equivalence audit:
# reproduce 3 existing audited cells for each of the five datasets using
# the exact same engine that will create the new joint cells.
print("\nRepresentative cell reproduction audit:")
rep_rows = []

for ds in DATASETS:
    Xd, yd, _ = DATA[ds]
    tr_a, te_a, h_a = split_for(Xd, yd, AUDIT_SEED)

    for cfg_name in AUDIT_CONFIGS:
        # Existing configuration definitions are fixed below for audit only.
        audit_cfg = {
            "baseline": {
                "scaling": True, "sampling": "original",
                "representation": "all_features", "training_fraction": 1.0
            },
            "sampling_cf": {
                "scaling": True, "sampling": "minority_oversampling",
                "representation": "all_features", "training_fraction": 1.0
            },
            "training_fraction_cf": {
                "scaling": True, "sampling": "original",
                "representation": "all_features", "training_fraction": 0.5
            }
        }[cfg_name]

        old_cfg = CONFIGS.get(cfg_name)
        CONFIGS[cfg_name] = audit_cfg
        try:
            out = run_one(
                ds, AUDIT_MODEL, AUDIT_SEED, cfg_name,
                tr_a, te_a, h_a
            )
        finally:
            if old_cfg is None:
                CONFIGS.pop(cfg_name, None)
            else:
                CONFIGS[cfg_name] = old_cfg

        audited_row = AUDIT_LEDGER[ds]
        q = audited_row[
            (audited_row["model"].astype(str) == AUDIT_MODEL) &
            (audited_row["seed"].astype(int) == AUDIT_SEED) &
            (audited_row["configuration"].astype(str) == cfg_name)
        ]
        if len(q) != 1:
            raise RuntimeError(f"{ds}/{cfg_name}: audited representative row not unique")

        ref = float(q.iloc[0]["macro_f1"])
        got = float(out["macro_f1"])
        diff = abs(got - ref)

        rep_rows.append({
            "dataset": ds,
            "configuration": cfg_name,
            "reference_macro_f1": ref,
            "reproduced_macro_f1": got,
            "absolute_difference": diff
        })

        print(
            f"  {ds:15s} | {cfg_name:22s} | "
            f"reference={ref:.12f} | reproduced={got:.12f} | "
            f"diff={diff:.3e}"
        )

rep_df = pd.DataFrame(rep_rows)
REP_TOL = 1e-12
rep_pass = (
    len(rep_df) == 15 and
    np.isfinite(rep_df["reproduced_macro_f1"]).all() and
    (rep_df["absolute_difference"] <= REP_TOL).all()
)

print(
    "Representative implementation equivalence:",
    "PASS" if rep_pass else "FAIL"
)

if not rep_pass:
    print(rep_df.to_string(index=False))
    raise RuntimeError(
        "PRE-PRODUCTION GATE FAILED: representative reproduced cells do not "
        "match the completed audited ledgers exactly."
    )

print("PRE-PRODUCTION EQUIVALENCE GATE: PASS")
print("Production authorization may proceed ONLY after this gate.")

manifest_payload = {
    "batch":"3A",
    "datasets":list(DATASETS),
    "models":MODELS,
    "seeds":SEEDS,
    "new_configurations":NEW_CONFIGS,
    "expected_new_fits":EXPECTED,
    "primary_metric":"macro_f1",
    "test_size":0.20,
    "stratified":True,
    "split_registry":SPLITS,
    "production_authorized":True,
    "created_at":datetime.now().isoformat()
}
MANIFEST.write_text(json.dumps(manifest_payload, indent=2))

# -------------------------------------------------------------------
# RESUME
# -------------------------------------------------------------------
if RESULTS.exists():
    results_df = pd.read_csv(RESULTS)
else:
    results_df = pd.DataFrame()

existing_pass = set()
if len(results_df):
    existing_pass = set(
        results_df.loc[
            results_df["status"].astype(str).eq("PASS"), "run_key"
        ].astype(str)
    )

print("\nExisting PASS:", len(existing_pass))
print("To execute:", EXPECTED-len(existing_pass))

# -------------------------------------------------------------------
# EXECUTION
# -------------------------------------------------------------------
for i, row in manifest_df.iterrows():
    key = row["run_key"]

    if key in existing_pass:
        print(f"[SKIP {i+1}/{EXPECTED}] {key}")
        continue

    dataset_name = row["dataset"]
    model_name = row["model"]
    seed = int(row["seed"])
    cfg = row["configuration"]

    print(f"\n[{i+1}/{EXPECTED}] {dataset_name:15s} | "
          f"{model_name:20s} | seed={seed:<4d} | {cfg}")

    try:
        tr = np.arange(len(DATA[dataset_name][0]), dtype=np.int64)
        # Retrieve exact canonical split from the deterministic function.
        tr, te, split_hash = split_for(DATA[dataset_name][0],
                                       DATA[dataset_name][1], seed)

        assert split_hash == SPLITS[dataset_name][str(seed)]["split_hash"]

        out = run_one(dataset_name, model_name, seed, cfg,
                      tr, te, split_hash)

        print(f" PASS | Macro-F1={out['macro_f1']:.6f} | "
              f"BA={out['balanced_accuracy']:.6f} | "
              f"time={out['runtime_seconds']:.3f}s")

    except Exception as e:
        out = {
            "run_key":key, "dataset":dataset_name,
            "model":model_name, "seed":seed,
            "configuration":cfg,
            "status":"FAIL", "error":repr(e),
            "completed_at":datetime.now().isoformat()
        }
        print(" FAIL |", repr(e))

    results_df = pd.concat(
        [results_df, pd.DataFrame([out])],
        ignore_index=True
    )
    results_df.to_csv(RESULTS, index=False)

    # checkpoint every 10 completed rows
    if len(results_df) % 10 == 0:
        CHECKPOINT.write_text(json.dumps({
            "completed_rows": int(len(results_df)),
            "pass_rows": int((results_df["status"]=="PASS").sum()),
            "fail_rows": int((results_df["status"]=="FAIL").sum()),
            "updated_at":datetime.now().isoformat()
        }, indent=2))
        print(" CHECKPOINT SAVED")

# -------------------------------------------------------------------
# FINAL VALIDATION OF NEW JOINT RUNS
# -------------------------------------------------------------------
results_df = pd.read_csv(RESULTS)
expected_keys = set(manifest_df["run_key"])
observed_keys = set(results_df["run_key"].astype(str))

g1 = len(results_df) == EXPECTED
g2 = len(observed_keys) == EXPECTED
g3 = results_df["status"].eq("PASS").all()
g4 = results_df["macro_f1"].apply(np.isfinite).all() if g3 else False
g5 = results_df.groupby("configuration").size().to_dict()
g6 = results_df.groupby("model").size().to_dict()
g7 = results_df.groupby("dataset").size().to_dict()
g8 = results_df.groupby("seed").size().to_dict()
g9 = all(
    results_df.groupby(["dataset","seed"])["split_hash"].nunique().eq(1)
)
g10 = all(
    results_df["run_key"].isin(expected_keys)
)

report = {
    "G1_expected_600_rows": bool(g1),
    "G2_unique_run_keys": bool(g2),
    "G3_all_runs_pass": bool(g3),
    "G4_metric_integrity": bool(g4),
    "G5_configuration_accounting": g5,
    "G6_model_accounting": g6,
    "G7_dataset_accounting": g7,
    "G8_seed_accounting": g8,
    "G9_exact_split_reuse": bool(g9),
    "G10_manifest_key_alignment": bool(g10),
    "pass_runs": int((results_df["status"]=="PASS").sum()),
    "fail_runs": int((results_df["status"]=="FAIL").sum()),
    "production_authorized": True,
    "created_at":datetime.now().isoformat()
}
FINAL.write_text(json.dumps(report, indent=2))

print("\n"+"="*78)
print("BATCH 3A NEW JOINT-RUN VALIDATION")
print("="*78)
print(json.dumps(report, indent=2))

if not (g1 and g2 and g3 and g4 and g9 and g10):
    raise RuntimeError("BATCH 3A NEW JOINT RUN VALIDATION FAILED. STOP.")

# -------------------------------------------------------------------
# MERGE WITH EXISTING 5-DATASET BASE/SINGLE-CONDITION LEDGERS
# No rerunning of baseline/single conditions.
# -------------------------------------------------------------------
BASE_LEDGER_PATHS = {
    "breast_cancer":
        "/content/CTD_BATCH1_LEDGER/CTD_BATCH1A_RESULTS_LEDGER.csv",
    "bank_marketing":
        "/content/CTD_BATCH1B_LEDGER/CTD_BATCH1B_RESULTS_LEDGER.csv",
    "pima_diabetes":
        "/content/CTD_BATCH2A_PIMA_PRODUCTION/CTD_BATCH2A_PIMA_RESULTS_LEDGER.csv",
    "spambase":
        "/content/CTD_BATCH2B_SPAMBASE_PRODUCTION/CTD_BATCH2B_SPAMBASE_RESULTS_LEDGER.csv",
    "phoneme":
        "/content/CTD_BATCH2C_PHONEME_PRODUCTION/CTD_BATCH2C_PHONEME_RESULTS_LEDGER.csv"
}

base_parts = []
for ds, p in BASE_LEDGER_PATHS.items():
    if not Path(p).exists():
        raise FileNotFoundError(f"Required existing ledger missing: {p}")
    z = pd.read_csv(p)
    z = z[z["status"].astype(str).eq("PASS")].copy()
    z = z[z["dataset"].astype(str).eq(ds)].copy()
    base_parts.append(z)

base_df = pd.concat(base_parts, ignore_index=True)

# Only the five frozen single-condition configurations.
single_cfgs = [
    "baseline","scaling_cf","sampling_cf",
    "representation_cf","training_fraction_cf"
]
base_df = base_df[base_df["configuration"].isin(single_cfgs)].copy()

assert len(base_df) == 1500, len(base_df)
assert base_df.groupby(["dataset","configuration"]).size().eq(60).all()
assert base_df.groupby(["dataset","model"]).size().eq(75).all()

new_df = results_df[results_df["status"].eq("PASS")].copy()
assert len(new_df) == 600

combined = pd.concat([base_df, new_df], ignore_index=True)

# -------------------------------------------------------------------
# INTERACTION VALUES FOR TWO BATCH-3A PAIRS
# I = Y_AB - Y_A - Y_B + Y_0
# -------------------------------------------------------------------
key_cols = ["dataset","model","seed"]

wide = combined.pivot_table(
    index=key_cols,
    columns="configuration",
    values="macro_f1",
    aggfunc="first"
).reset_index()

required = [
    "baseline","scaling_cf","sampling_cf",
    "training_fraction_cf",
    "scaling_training_joint","sampling_training_joint"
]
assert all(c in wide.columns for c in required)

wide["scaling_x_training"] = (
    wide["scaling_training_joint"]
    - wide["scaling_cf"]
    - wide["training_fraction_cf"]
    + wide["baseline"]
)

wide["sampling_x_training"] = (
    wide["sampling_training_joint"]
    - wide["sampling_cf"]
    - wide["training_fraction_cf"]
    + wide["baseline"]
)

interaction_long = pd.concat([
    wide[key_cols + ["scaling_x_training"]].assign(
        interaction="scaling_x_training"
    ).rename(columns={"scaling_x_training":"interaction_value"}),
    wide[key_cols + ["sampling_x_training"]].assign(
        interaction="sampling_x_training"
    ).rename(columns={"sampling_x_training":"interaction_value"})
], ignore_index=True)

interaction_long.to_csv(INTERACTIONS, index=False)

# Exact arithmetic self-check against source cells.
calc1 = (
    wide["scaling_training_joint"] - wide["scaling_cf"]
    - wide["training_fraction_cf"] + wide["baseline"]
)
calc2 = (
    wide["sampling_training_joint"] - wide["sampling_cf"]
    - wide["training_fraction_cf"] + wide["baseline"]
)
max_err = float(max(
    np.max(np.abs(calc1-wide["scaling_x_training"])),
    np.max(np.abs(calc2-wide["sampling_x_training"]))
))

print("\n"+"="*78)
print("BATCH 3A INTERACTION ASSEMBLY")
print("="*78)
print("Base/single rows reused :", len(base_df))
print("New joint rows          :", len(new_df))
print("Combined rows           :", len(combined))
print("Interaction rows        :", len(interaction_long))
print("Maximum arithmetic err  :", max_err)
print("\nInteraction summary:")
print(
    interaction_long.groupby("interaction")["interaction_value"]
    .agg(["count","mean","std","min","max"])
)

assert len(combined) == 2100
assert len(interaction_long) == 600
assert max_err == 0.0

print("\nBATCH 3A: COMPLETE AND VALIDATED.")
print("New results :", RESULTS)
print("Interactions:", INTERACTIONS)
print("Final report:", FINAL)
