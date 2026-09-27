# CTD IJDSA REVISION — BATCH 2A PIMA PRODUCTION v1
# 300 fits: 15 seeds × 4 models × 5 configurations.
# Pima only. Explicit OpenML ID 37. Checkpoint/resume enabled.
# DO NOT modify the frozen protocol before execution.

import gc, json, time, hashlib, platform, warnings
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
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score

BASE = Path("/content/CTD_BATCH2A_PIMA_PRODUCTION")
BASE.mkdir(parents=True, exist_ok=True)

DATASET="pima_diabetes"
OPENML_ID=37
SEEDS=[42,123,2026,7,19,31,53,71,89,107,149,173,211,257,311]
MODELS=["logistic_regression","decision_tree","random_forest","linear_svm"]
CONFIGS={
"baseline":{"scaling":True,"sampling":"original","representation":"all_features","training_fraction":1.0},
"scaling_cf":{"scaling":False,"sampling":"original","representation":"all_features","training_fraction":1.0},
"sampling_cf":{"scaling":True,"sampling":"minority_oversampling","representation":"all_features","training_fraction":1.0},
"representation_cf":{"scaling":True,"sampling":"original","representation":"pca95","training_fraction":1.0},
"training_fraction_cf":{"scaling":True,"sampling":"original","representation":"all_features","training_fraction":0.5}}
EXPECTED=len(SEEDS)*len(MODELS)*len(CONFIGS)

print("="*78)
print("CTD IJDSA REVISION — BATCH 2A PIMA PRODUCTION v1")
print("="*78)
print("Expected fits:",EXPECTED)
print("Python:",platform.python_version()," NumPy:",np.__version__," sklearn:",sklearn.__version__)

# Explicit dataset identity
ds=fetch_openml(data_id=OPENML_ID,as_frame=True)
X=pd.DataFrame(ds.data).copy()
y=pd.Series(ds.target).astype(str).copy()
for c in X.columns: X[c]=pd.to_numeric(X[c],errors="coerce")
mask=~X.isna().any(axis=1)
X=X.loc[mask].reset_index(drop=True); y=y.loc[mask].reset_index(drop=True)

assert X.shape==(768,8),X.shape
assert y.nunique()==2
assert X.isna().sum().sum()==0

def sha_df(df):
    return hashlib.sha256(df.to_csv(index=False,lineterminator="\n").encode()).hexdigest()
def sha_y(s):
    return hashlib.sha256(s.astype(str).to_csv(index=False,header=True,lineterminator="\n").encode()).hexdigest()
feature_sha=sha_df(X); target_sha=sha_y(y)
hh=hashlib.sha256(); hh.update(feature_sha.encode()); hh.update(target_sha.encode())
combined_sha=hh.hexdigest()

print("\nPima:",X.shape,"classes:",y.nunique(),"duplicates:",X.duplicated().sum())
print("Feature SHA:",feature_sha)
print("Target SHA :",target_sha)
print("Combined SHA:",combined_sha)

def split_hash(a,b):
    h=hashlib.sha256()
    h.update(np.asarray(a,dtype=np.int64).tobytes())
    h.update(np.asarray(b,dtype=np.int64).tobytes())
    return h.hexdigest()

def make_split(seed):
    idx=np.arange(len(X),dtype=np.int64)
    a,b=train_test_split(idx,test_size=.20,stratify=y,random_state=seed)
    return np.asarray(a,dtype=np.int64),np.asarray(b,dtype=np.int64),split_hash(a,b)

splits={}
for s in SEEDS:
    a,b,h=make_split(s)
    splits[str(s)]={"train_size":len(a),"test_size":len(b),"split_hash":h}
print("\nCanonical splits:")
for s,v in splits.items(): print(s,v)

def model_factory(name,seed):
    if name=="logistic_regression":
        return LogisticRegression(max_iter=2000,solver="liblinear",random_state=seed)
    if name=="decision_tree":
        return DecisionTreeClassifier(random_state=seed)
    if name=="random_forest":
        return RandomForestClassifier(n_estimators=200,random_state=seed,n_jobs=-1)
    if name=="linear_svm":
        return LinearSVC(C=1.0,penalty="l2",loss="squared_hinge",
                         dual=False,tol=1e-3,max_iter=1000,random_state=seed)
    raise ValueError(name)

def oversample(A,b,seed):
    rng=np.random.RandomState(seed); yv=np.asarray(b)
    classes,counts=np.unique(yv,return_counts=True); mx=int(counts.max())
    Ax=[np.asarray(A)]; by=[yv]
    for cls,n in zip(classes,counts):
        d=mx-int(n)
        if d>0:
            ids=np.where(yv==cls)[0]
            take=rng.choice(ids,size=d,replace=True)
            Ax.append(np.asarray(A)[take]); by.append(yv[take])
    A=np.vstack(Ax); b=np.concatenate(by); order=rng.permutation(len(b))
    return A[order],b[order]

def run_one(model_name,seed,cname,train_idx,test_idx,split_id):
    cfg=CONFIGS[cname]; t0=time.time()
    A=X.iloc[train_idx].to_numpy(float); B=X.iloc[test_idx].to_numpy(float)
    ya=y.iloc[train_idx].to_numpy(); yb=y.iloc[test_idx].to_numpy()
    if cfg["training_fraction"]<1:
        A,_,ya,_=train_test_split(A,ya,train_size=.5,stratify=ya,random_state=seed)
    if cfg["scaling"]:
        sc=StandardScaler(); A=sc.fit_transform(A); B=sc.transform(B)
    pc=None; pv=None
    if cfg["representation"]=="pca95":
        p=PCA(n_components=.95,svd_solver="full",random_state=seed)
        A=p.fit_transform(A); B=p.transform(B); pc=A.shape[1]; pv=float(p.explained_variance_ratio_.sum())
    if cfg["sampling"]=="minority_oversampling": A,ya=oversample(A,ya,seed)
    m=model_factory(model_name,seed)
    with warnings.catch_warnings(record=True) as ww:
        warnings.simplefilter("always"); m.fit(A,ya)
    pred=m.predict(B); elapsed=time.time()-t0
    return {"run_key":f"{DATASET}__{model_name}__seed{seed}__{cname}",
            "dataset":DATASET,"openml_data_id":OPENML_ID,"model":model_name,"seed":seed,
            "configuration":cname,"scaling":cfg["scaling"],"sampling":cfg["sampling"],
            "representation":cfg["representation"],"training_fraction":cfg["training_fraction"],
            "split_hash":split_id,"train_rows":len(A),"test_rows":len(B),
            "pca_components":pc,"pca_explained_variance":pv,
            "macro_f1":float(f1_score(yb,pred,average="macro")),
            "accuracy":float(accuracy_score(yb,pred)),
            "balanced_accuracy":float(balanced_accuracy_score(yb,pred)),
            "runtime_seconds":elapsed,"warnings":json.dumps([str(w.message) for w in ww]),
            "status":"PASS","error":"","completed_at":datetime.now().isoformat()}

results_path=BASE/"CTD_BATCH2A_PIMA_RESULTS_LEDGER.csv"
status_path=BASE/"CTD_BATCH2A_PIMA_RUN_STATUS.csv"
checkpoint_path=BASE/"CTD_BATCH2A_PIMA_CHECKPOINT.json"
manifest_path=BASE/"CTD_BATCH2A_PIMA_PRODUCTION_MANIFEST.json"

manifest={"batch":"2A","dataset":DATASET,"openml_data_id":OPENML_ID,
          "shape":list(X.shape),"feature_sha256":feature_sha,"target_sha256":target_sha,
          "combined_sha256":combined_sha,"seeds":SEEDS,"models":MODELS,
          "configurations":CONFIGS,"expected_runs":EXPECTED,"primary_metric":"macro_f1",
          "test_size":.20,"stratified":True,"split_registry":splits,
          "production_authorized":True,"created_at":datetime.now().isoformat()}
manifest_path.write_text(json.dumps(manifest,indent=2))

if results_path.exists():
    results=pd.read_csv(results_path)
    done=set(results.loc[results.status=="PASS","run_key"].astype(str))
else:
    results=pd.DataFrame(); done=set()

print("\nStarting/resuming production. Existing PASS:",len(done))

counter=0
for seed in SEEDS:
    tr,te,sh=make_split(seed)
    for model in MODELS:
        for cfg in CONFIGS:
            counter+=1
            key=f"{DATASET}__{model}__seed{seed}__{cfg}"
            if key in done:
                print(f"[{counter}/{EXPECTED}] SKIP {key}"); continue
            print(f"\n[{counter}/{EXPECTED}] {model} | seed={seed} | {cfg}")
            try:
                r=run_one(model,seed,cfg,tr,te,sh)
                print(f"PASS | F1={r['macro_f1']:.6f} | BA={r['balanced_accuracy']:.6f} | {r['runtime_seconds']:.3f}s")
            except Exception as e:
                r={"run_key":key,"dataset":DATASET,"openml_data_id":OPENML_ID,
                   "model":model,"seed":seed,"configuration":cfg,"split_hash":sh,
                   "status":"FAIL","error":f"{type(e).__name__}: {e}",
                   "completed_at":datetime.now().isoformat()}
                print("FAIL |",r["error"])
            results=pd.concat([results,pd.DataFrame([r])],ignore_index=True)
            results.to_csv(results_path,index=False)
            results[["run_key","dataset","model","seed","configuration","split_hash","status","error"]].to_csv(status_path,index=False)
            checkpoint={"expected":EXPECTED,"pass":int((results.status=="PASS").sum()),
                        "fail":int((results.status=="FAIL").sum()),
                        "remaining":EXPECTED-int((results.status=="PASS").sum()),
                        "last_run_key":key,"updated_at":datetime.now().isoformat()}
            checkpoint_path.write_text(json.dumps(checkpoint,indent=2))
            gc.collect()

# Final validation
df=pd.read_csv(results_path)
g1=len(df)==EXPECTED
g2=df.run_key.duplicated().sum()==0
g3=(df.status=="PASS").all()
g4=set(df.loc[df.status=="PASS","configuration"])==set(CONFIGS) and all(df.loc[df.status=="PASS","configuration"].value_counts()==60)
g5=set(df.loc[df.status=="PASS","model"])==set(MODELS) and all(df.loc[df.status=="PASS","model"].value_counts()==75)
g6=set(df.loc[df.status=="PASS","seed"])==set(SEEDS) and all(df.loc[df.status=="PASS","seed"].value_counts()==20)
vals=df[["macro_f1","accuracy","balanced_accuracy"]].to_numpy(float)
g7=np.isfinite(vals).all() and (vals>=0).all() and (vals<=1).all()
g8=df.groupby("seed").split_hash.nunique().eq(1).all() and len(df.groupby("seed").split_hash.nunique())==15
g9=df.groupby(["seed","split_hash"]).size().eq(20).all() and len(df.groupby(["seed","split_hash"]))==15
g10=df.dataset.nunique()==1 and df.dataset.iloc[0]==DATASET and df.openml_data_id.nunique()==1 and int(df.openml_data_id.iloc[0])==OPENML_ID
expected_keys={f"{DATASET}__{m}__seed{s}__{c}" for s in SEEDS for m in MODELS for c in CONFIGS}
g11=set(df.run_key)==expected_keys
gates=[g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11]

report={"batch":"2A","dataset":DATASET,"expected_runs":EXPECTED,"actual_rows":len(df),
        "pass_runs":int((df.status=="PASS").sum()),"fail_runs":int((df.status=="FAIL").sum()),
        "gates":{f"G{i+1}":bool(v) for i,v in enumerate(gates)},
        "final_validation":"PASS" if all(gates) else "HOLD",
        "timestamp":datetime.now().isoformat()}
report_path=BASE/"CTD_BATCH2A_PIMA_FINAL_VALIDATION_REPORT.json"
report_path.write_text(json.dumps(report,indent=2))

print("\n"+"="*78)
print("BATCH 2A PIMA PRODUCTION — FINAL")
print("="*78)
for i,v in enumerate(gates,1): print(f"G{i}: {'PASS' if v else 'FAIL'}")
print("PASS runs:",int((df.status=="PASS").sum())," FAIL runs:",int((df.status=="FAIL").sum()))
print("FINAL:",report["final_validation"])
print("Results:",results_path)
print("Status:",status_path)
print("Checkpoint:",checkpoint_path)
print("Manifest:",manifest_path)
print("Report:",report_path)
print("="*78)
