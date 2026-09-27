#!/usr/bin/env python3
"""CTD IJDSA Major Revision — BATCH 0 ENGINE VALIDATION v1.0
Synthetic known-answer tests only; no real benchmark is executed.
"""
from itertools import product
from math import sqrt
from statistics import mean, stdev

DATASETS = ["breast_cancer","bank_marketing","covertype","pima_diabetes","spambase","phoneme"]
MODELS = ["logistic_regression","decision_tree","random_forest","linear_svm"]
SEEDS = [42,123,2026,7,19,31,53,71,89,107,149,173,211,257,311]
DECISIONS = {
    "scaling": ("standard_scaler","no_scaling"),
    "sampling": ("original","minority_oversampling"),
    "representation": ("all_features","pca95"),
    "training_fraction": ("100_percent","50_percent"),
}
INTERACTION_PAIRS = [
    ("scaling","training_fraction"),
    ("sampling","training_fraction"),
    ("sampling","representation"),
    ("representation","training_fraction"),
]
CANDIDATE_THRESHOLDS = [0.005,0.01,0.02]
PRIMARY_THRESHOLD = 0.01
EXPECTED_FITS = len(DATASETS)*len(MODELS)*len(SEEDS)*(1+len(DECISIONS)+len(INTERACTION_PAIRS))

def cde(baseline, counterfactual):
    return baseline-counterfactual

def interaction(y0, yi, yj, yij):
    return yij-yi-yj+y0

def raw_reversal(g0, gcf):
    return (g0*gcf)<0

def practical_reversal(g0, gcf, delta):
    return raw_reversal(g0,gcf) and abs(gcf-g0)>=delta

def mean_ci95(values):
    xs=list(values)
    if len(xs)<2: return mean(xs), float("nan"), float("nan")
    m=mean(xs); se=stdev(xs)/sqrt(len(xs)); h=1.96*se
    return m,m-h,m+h

def test_cde():
    assert abs(cde(0.80,0.75)-0.05)<1e-12
    assert abs(cde(0.75,0.80)+0.05)<1e-12

def test_interaction():
    assert abs(interaction(0.80,0.75,0.78,0.73))<1e-12
    assert abs(interaction(0.80,0.75,0.78,0.70)+0.03)<1e-12

def test_reversal_threshold():
    assert raw_reversal(0.020,-0.010)
    assert practical_reversal(0.020,-0.010,0.01)
    assert raw_reversal(0.002,-0.003)
    assert not practical_reversal(0.002,-0.003,0.01)
    assert not raw_reversal(0.020,0.010)

def test_run_accounting():
    keys=[]
    for d,m,s in product(DATASETS,MODELS,SEEDS):
        keys.append((d,m,s,"baseline"))
        for decision in DECISIONS:
            keys.append((d,m,s,decision))
        for a,b in INTERACTION_PAIRS:
            keys.append((d,m,s,f"{a}__{b}"))
    assert len(keys)==3240
    assert len(set(keys))==3240

def test_design():
    assert len(DATASETS)==6 and len(MODELS)==4 and len(SEEDS)==15
    assert len(DECISIONS)==4 and len(INTERACTION_PAIRS)==4
    assert ("scaling","representation") not in INTERACTION_PAIRS
    assert PRIMARY_THRESHOLD in CANDIDATE_THRESHOLDS

def main():
    tests=[("CDE",test_cde),("interaction",test_interaction),
           ("ranking threshold",test_reversal_threshold),
           ("run accounting",test_run_accounting),("design",test_design)]
    failed=[]
    for name,fn in tests:
        try: fn(); print(f"[PASS] {name}")
        except Exception as e: failed.append((name,e)); print(f"[FAIL] {name}: {e}")
    print(f"Expected production fits: {EXPECTED_FITS}")
    if failed: raise SystemExit(1)
    print("ALL BATCH 0 ENGINE TESTS PASSED.")
    print("No real dataset experiments were executed.")

if __name__=="__main__":
    main()
