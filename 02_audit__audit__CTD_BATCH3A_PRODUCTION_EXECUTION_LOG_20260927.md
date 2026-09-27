```
==============================================================================
CTD IJDSA REVISION
BATCH 3A v4 — EXACT-ENGINE PRE-PRODUCTION EQUIVALENCE AUDIT
==============================================================================
Script : /content/CTD_BATCH3A_SCALING_SAMPLING_TRAINING_PRODUCTION_v4_EXACT_ENGINE_GATED.py
Size   : 30022 bytes

Bank Marketing : Batch 1B-R1 authoritative evidence
Historical Bank ledger: EXCLUDED
Breast Cancer  : recovered evidence with documented tolerance
Pima/Spambase/Phoneme: strict equivalence

NO 600-RUN PRODUCTION WILL START UNLESS
THE COMPLETE EQUIVALENCE GATE PASSES.
==============================================================================
==============================================================================
CTD IJDSA REVISION — BATCH 3A PRODUCTION v1.0
==============================================================================
Expected NEW fits: 600
Python: 3.13.15
NumPy: 2.1.3 Pandas: 2.2.3 Scikit-learn: 1.6.1
Datasets: ['breast_cancer', 'bank_marketing', 'pima_diabetes', 'spambase', 'phoneme']
Models: ['logistic_regression', 'decision_tree', 'random_forest', 'linear_svm']
Seeds: [42, 123, 2026, 7, 19, 31, 53, 71, 89, 107, 149, 173, 211, 257, 311]
New configurations: ['scaling_training_joint', 'sampling_training_joint']

Loading: breast_cancer
 shape: (569, 30) classes: 2 duplicates: 0

Loading: bank_marketing
 shape: (45211, 16) classes: 2 duplicates: 0

Loading: pima_diabetes
 shape: (768, 8) classes: 2 duplicates: 0

Loading: spambase
 shape: (4601, 57) classes: 2 duplicates: 394

Loading: phoneme
 shape: (5404, 5) classes: 2 duplicates: 9

==============================================================================
BATCH 3A PRE-PRODUCTION EQUIVALENCE AUDIT
==============================================================================
Exact canonical split identity: PASS

Representative cell reproduction audit:
  breast_cancer   | baseline               | reference=0.981151000000 | reproduced=0.981150793651 | diff=2.063e-07
  breast_cancer   | sampling_cf            | reference=0.944481000000 | reproduced=0.944480519481 | diff=4.805e-07
  breast_cancer   | training_fraction_cf   | reference=0.953534000000 | reproduced=0.953533871362 | diff=1.286e-07
  bank_marketing  | baseline               | reference=0.699448789529 | reproduced=0.699448789529 | diff=0.000e+00
  bank_marketing  | sampling_cf            | reference=0.728793479490 | reproduced=0.728793479490 | diff=0.000e+00
  bank_marketing  | training_fraction_cf   | reference=0.701756069187 | reproduced=0.701756069187 | diff=0.000e+00
  pima_diabetes   | baseline               | reference=0.674230769231 | reproduced=0.674230769231 | diff=0.000e+00
  pima_diabetes   | sampling_cf            | reference=0.717456481854 | reproduced=0.717456481854 | diff=0.000e+00
  pima_diabetes   | training_fraction_cf   | reference=0.674230769231 | reproduced=0.674230769231 | diff=0.000e+00
  spambase        | baseline               | reference=0.925783356578 | reproduced=0.925783356578 | diff=1.110e-16
  spambase        | sampling_cf            | reference=0.918462703745 | reproduced=0.918462703745 | diff=1.110e-16
  spambase        | training_fraction_cf   | reference=0.917664938316 | reproduced=0.917664938316 | diff=0.000e+00
  phoneme         | baseline               | reference=0.654568694394 | reproduced=0.654568694394 | diff=0.000e+00
  phoneme         | sampling_cf            | reference=0.696119935363 | reproduced=0.696119935363 | diff=0.000e+00
  phoneme         | training_fraction_cf   | reference=0.660199304441 | reproduced=0.660199304441 | diff=0.000e+00
Representative implementation equivalence: PASS
PRE-PRODUCTION EQUIVALENCE GATE: PASS
Production authorization may proceed ONLY after this gate.
Bank Marketing source: Batch 1B-R1 authoritative evidence.

Existing PASS: 0
To execute: 600

[1/600] breast_cancer   | logistic_regression  | seed=42   | scaling_training_joint
 PASS | Macro-F1=0.932981 | BA=0.926587 | time=0.007s

[2/600] breast_cancer   | logistic_regression  | seed=42   | sampling_training_joint
 PASS | Macro-F1=0.953534 | BA=0.960317 | time=0.007s

[3/600] breast_cancer   | decision_tree        | seed=42   | scaling_training_joint
 PASS | Macro-F1=0.907468 | BA=0.915675 | time=0.008s

[4/600] breast_cancer   | decision_tree        | seed=42   | sampling_training_joint
 PASS | Macro-F1=0.916361 | BA=0.922619 | time=0.009s

[5/600] breast_cancer   | random_forest        | seed=42   | scaling_training_joint
 PASS | Macro-F1=0.943452 | BA=0.943452 | time=0.478s

[6/600] breast_cancer   | random_forest        | seed=42   | sampling_training_joint
 PASS | Macro-F1=0.925319 | BA=0.929563 | time=0.550s

[7/600] breast_cancer   | linear_svm           | seed=42   | scaling_training_joint
 PASS | Macro-F1=0.942867 | BA=0.938492 | time=0.004s

[8/600] breast_cancer   | linear_svm           | seed=42   | sampling_training_joint
 PASS | Macro-F1=0.897775 | BA=0.903770 | time=0.006s

[9/600] breast_cancer   | logistic_regression  | seed=123  | scaling_training_joint
 PASS | Macro-F1=0.953106 | BA=0.955357 | time=0.007s

[10/600] breast_cancer   | logistic_regression  | seed=123  | sampling_training_joint
 PASS | Macro-F1=0.962302 | BA=0.962302 | time=0.006s
 CHECKPOINT SAVED

[11/600] breast_cancer   | decision_tree        | seed=123  | scaling_training_joint
 PASS | Macro-F1=0.935494 | BA=0.946429 | time=0.008s

[12/600] breast_cancer   | decision_tree        | seed=123  | sampling_training_joint
 PASS | Macro-F1=0.916361 | BA=0.922619 | time=0.009s

[13/600] breast_cancer   | random_forest        | seed=123  | scaling_training_joint
 PASS | Macro-F1=0.972120 | BA=0.979167 | time=0.492s

[14/600] breast_cancer   | random_forest        | seed=123  | sampling_training_joint
 PASS | Macro-F1=0.981330 | BA=0.986111 | time=0.523s

[15/600] breast_cancer   | linear_svm           | seed=123  | scaling_training_joint
 PASS | Macro-F1=0.981151 | BA=0.981151 | time=0.016s

[16/600] breast_cancer   | linear_svm           | seed=123  | sampling_training_joint
 PASS | Macro-F1=0.934348 | BA=0.936508 | time=0.008s

[17/600] breast_cancer   | logistic_regression  | seed=2026 | scaling_training_joint
 PASS | Macro-F1=0.925319 | BA=0.929563 | time=0.006s

[18/600] breast_cancer   | logistic_regression  | seed=2026 | sampling_training_joint
 PASS | Macro-F1=0.962660 | BA=0.967262 | time=0.007s

[19/600] breast_cancer   | decision_tree        | seed=2026 | scaling_training_joint
 PASS | Macro-F1=0.925319 | BA=0.929563 | time=0.006s

[20/600] breast_cancer   | decision_tree        | seed=2026 | sampling_training_joint
 PASS | Macro-F1=0.935494 | BA=0.946429 | time=0.009s
 CHECKPOINT SAVED

[21/600] breast_cancer   | random_forest        | seed=2026 | scaling_training_joint
 PASS | Macro-F1=0.943990 | BA=0.948413 | time=0.495s

[22/600] breast_cancer   | random_forest        | seed=2026 | sampling_training_joint
 PASS | Macro-F1=0.944481 | BA=0.953373 | time=0.496s

[23/600] breast_cancer   | linear_svm           | seed=2026 | scaling_training_joint
 PASS | Macro-F1=0.934348 | BA=0.936508 | time=0.005s

[24/600] breast_cancer   | linear_svm           | seed=2026 | sampling_training_joint
 PASS | Macro-F1=0.971863 | BA=0.974206 | time=0.006s

[25/600] breast_cancer   | logistic_regression  | seed=7    | scaling_training_joint
 PASS | Macro-F1=0.972120 | BA=0.979167 | time=0.007s

[26/600] breast_cancer   | logistic_regression  | seed=7    | sampling_training_joint
 PASS | Macro-F1=0.971863 | BA=0.974206 | time=0.007s

[27/600] breast_cancer   | decision_tree        | seed=7    | scaling_training_joint
 PASS | Macro-F1=0.890665 | BA=0.906746 | time=0.007s

[28/600] breast_cancer   | decision_tree        | seed=7    | sampling_training_joint
 PASS | Macro-F1=0.917064 | BA=0.927579 | time=0.008s

[29/600] breast_cancer   | random_forest        | seed=7    | scaling_training_joint
 PASS | Macro-F1=0.962660 | BA=0.967262 | time=0.465s

[30/600] breast_cancer   | random_forest        | seed=7    | sampling_training_joint
 PASS | Macro-F1=0.944481 | BA=0.953373 | time=0.508s
 CHECKPOINT SAVED

[31/600] breast_cancer   | linear_svm           | seed=7    | scaling_training_joint
 PASS | Macro-F1=0.972120 | BA=0.979167 | time=0.005s

[32/600] breast_cancer   | linear_svm           | seed=7    | sampling_training_joint
 PASS | Macro-F1=0.962987 | BA=0.972222 | time=0.005s

[33/600] breast_cancer   | logistic_regression  | seed=19   | scaling_training_joint
 PASS | Macro-F1=0.980956 | BA=0.976190 | time=0.006s

[34/600] breast_cancer   | logistic_regression  | seed=19   | sampling_training_joint
 PASS | Macro-F1=0.980956 | BA=0.976190 | time=0.006s

[35/600] breast_cancer   | decision_tree        | seed=19   | scaling_training_joint
 PASS | Macro-F1=0.952638 | BA=0.950397 | time=0.007s

[36/600] breast_cancer   | decision_tree        | seed=19   | sampling_training_joint
 PASS | Macro-F1=0.971583 | BA=0.969246 | time=0.009s

[37/600] breast_cancer   | random_forest        | seed=19   | scaling_training_joint
 PASS | Macro-F1=0.990528 | BA=0.988095 | time=0.485s

[38/600] breast_cancer   | random_forest        | seed=19   | sampling_training_joint
 PASS | Macro-F1=0.990528 | BA=0.988095 | time=0.581s

[39/600] breast_cancer   | linear_svm           | seed=19   | scaling_training_joint
 PASS | Macro-F1=0.980956 | BA=0.976190 | time=0.006s

[40/600] breast_cancer   | linear_svm           | seed=19   | sampling_training_joint
 PASS | Macro-F1=0.961486 | BA=0.952381 | time=0.006s
 CHECKPOINT SAVED

[41/600] breast_cancer   | logistic_regression  | seed=31   | scaling_training_joint
 PASS | Macro-F1=0.922973 | BA=0.914683 | time=0.007s

[42/600] breast_cancer   | logistic_regression  | seed=31   | sampling_training_joint
 PASS | Macro-F1=0.913832 | BA=0.907738 | time=0.007s

[43/600] breast_cancer   | decision_tree        | seed=31   | scaling_training_joint
 PASS | Macro-F1=0.904778 | BA=0.900794 | time=0.008s

[44/600] breast_cancer   | decision_tree        | seed=31   | sampling_training_joint
 PASS | Macro-F1=0.894684 | BA=0.888889 | time=0.010s

[45/600] breast_cancer   | random_forest        | seed=31   | scaling_training_joint
 PASS | Macro-F1=0.952129 | BA=0.945437 | time=0.481s

[46/600] breast_cancer   | random_forest        | seed=31   | sampling_training_joint
 PASS | Macro-F1=0.932206 | BA=0.921627 | time=0.494s

[47/600] breast_cancer   | linear_svm           | seed=31   | scaling_training_joint
 PASS | Macro-F1=0.932981 | BA=0.926587 | time=0.005s

[48/600] breast_cancer   | linear_svm           | seed=31   | sampling_training_joint
 PASS | Macro-F1=0.912837 | BA=0.902778 | time=0.006s

[49/600] breast_cancer   | logistic_regression  | seed=53   | scaling_training_joint
 PASS | Macro-F1=0.932981 | BA=0.926587 | time=0.006s

[50/600] breast_cancer   | logistic_regression  | seed=53   | sampling_training_joint
 PASS | Macro-F1=1.000000 | BA=1.000000 | time=0.007s
 CHECKPOINT SAVED

[51/600] breast_cancer   | decision_tree        | seed=53   | scaling_training_joint
 PASS | Macro-F1=0.933693 | BA=0.931548 | time=0.006s

[52/600] breast_cancer   | decision_tree        | seed=53   | sampling_training_joint
 PASS | Macro-F1=0.943990 | BA=0.948413 | time=0.009s

[53/600] breast_cancer   | random_forest        | seed=53   | scaling_training_joint
 PASS | Macro-F1=0.962660 | BA=0.967262 | time=0.464s

[54/600] breast_cancer   | random_forest        | seed=53   | sampling_training_joint
 PASS | Macro-F1=0.972120 | BA=0.979167 | time=0.512s

[55/600] breast_cancer   | linear_svm           | seed=53   | scaling_training_joint
 PASS | Macro-F1=0.912837 | BA=0.902778 | time=0.005s

[56/600] breast_cancer   | linear_svm           | seed=53   | sampling_training_joint
 PASS | Macro-F1=0.980956 | BA=0.976190 | time=0.007s

[57/600] breast_cancer   | logistic_regression  | seed=71   | scaling_training_joint
 PASS | Macro-F1=0.923822 | BA=0.919643 | time=0.006s

[58/600] breast_cancer   | logistic_regression  | seed=71   | sampling_training_joint
 PASS | Macro-F1=0.971583 | BA=0.969246 | time=0.007s

[59/600] breast_cancer   | decision_tree        | seed=71   | scaling_training_joint
 PASS | Macro-F1=0.866689 | BA=0.863095 | time=0.007s

[60/600] breast_cancer   | decision_tree        | seed=71   | sampling_training_joint
 PASS | Macro-F1=0.916361 | BA=0.922619 | time=0.009s
 CHECKPOINT SAVED

[61/600] breast_cancer   | random_forest        | seed=71   | scaling_training_joint
 PASS | Macro-F1=0.895804 | BA=0.893849 | time=0.684s

[62/600] breast_cancer   | random_forest        | seed=71   | sampling_training_joint
 PASS | Macro-F1=0.895804 | BA=0.893849 | time=0.754s

[63/600] breast_cancer   | linear_svm           | seed=71   | scaling_training_joint
 PASS | Macro-F1=0.923822 | BA=0.919643 | time=0.008s

[64/600] breast_cancer   | linear_svm           | seed=71   | sampling_training_joint
 PASS | Macro-F1=0.953106 | BA=0.955357 | time=0.010s

[65/600] breast_cancer   | logistic_regression  | seed=89   | scaling_training_joint
 PASS | Macro-F1=0.961911 | BA=0.957341 | time=0.012s

[66/600] breast_cancer   | logistic_regression  | seed=89   | sampling_training_joint
 PASS | Macro-F1=0.961911 | BA=0.957341 | time=0.006s

[67/600] breast_cancer   | decision_tree        | seed=89   | scaling_training_joint
 PASS | Macro-F1=0.952638 | BA=0.950397 | time=0.007s

[68/600] breast_cancer   | decision_tree        | seed=89   | sampling_training_joint
 PASS | Macro-F1=0.915590 | BA=0.917659 | time=0.017s

[69/600] breast_cancer   | random_forest        | seed=89   | scaling_training_joint
 PASS | Macro-F1=0.942867 | BA=0.938492 | time=0.635s

[70/600] breast_cancer   | random_forest        | seed=89   | sampling_training_joint
 PASS | Macro-F1=0.953534 | BA=0.960317 | time=0.586s
 CHECKPOINT SAVED

[71/600] breast_cancer   | linear_svm           | seed=89   | scaling_training_joint
 PASS | Macro-F1=0.971583 | BA=0.969246 | time=0.011s

[72/600] breast_cancer   | linear_svm           | seed=89   | sampling_training_joint
 PASS | Macro-F1=0.961911 | BA=0.957341 | time=0.014s

[73/600] breast_cancer   | logistic_regression  | seed=107  | scaling_training_joint
 PASS | Macro-F1=0.953106 | BA=0.955357 | time=0.012s

[74/600] breast_cancer   | logistic_regression  | seed=107  | sampling_training_joint
 PASS | Macro-F1=0.962660 | BA=0.967262 | time=0.012s

[75/600] breast_cancer   | decision_tree        | seed=107  | scaling_training_joint
 PASS | Macro-F1=0.926570 | BA=0.939484 | time=0.016s

[76/600] breast_cancer   | decision_tree        | seed=107  | sampling_training_joint
 PASS | Macro-F1=0.944481 | BA=0.953373 | time=0.017s

[77/600] breast_cancer   | random_forest        | seed=107  | scaling_training_joint
 PASS | Macro-F1=0.971863 | BA=0.974206 | time=0.548s

[78/600] breast_cancer   | random_forest        | seed=107  | sampling_training_joint
 PASS | Macro-F1=0.962660 | BA=0.967262 | time=0.481s

[79/600] breast_cancer   | linear_svm           | seed=107  | scaling_training_joint
 PASS | Macro-F1=0.953106 | BA=0.955357 | time=0.005s

[80/600] breast_cancer   | linear_svm           | seed=107  | sampling_training_joint
 PASS | Macro-F1=0.962302 | BA=0.962302 | time=0.006s
 CHECKPOINT SAVED

[81/600] breast_cancer   | logistic_regression  | seed=149  | scaling_training_joint
 PASS | Macro-F1=0.922973 | BA=0.914683 | time=0.007s

[82/600] breast_cancer   | logistic_regression  | seed=149  | sampling_training_joint
 PASS | Macro-F1=0.961486 | BA=0.952381 | time=0.007s

[83/600] breast_cancer   | decision_tree        | seed=149  | scaling_training_joint
 PASS | Macro-F1=0.922051 | BA=0.909722 | time=0.007s

[84/600] breast_cancer   | decision_tree        | seed=149  | sampling_training_joint
 PASS | Macro-F1=0.922051 | BA=0.909722 | time=0.009s

[85/600] breast_cancer   | random_forest        | seed=149  | scaling_training_joint
 PASS | Macro-F1=0.931367 | BA=0.916667 | time=0.469s

[86/600] breast_cancer   | random_forest        | seed=149  | sampling_training_joint
 PASS | Macro-F1=0.910588 | BA=0.892857 | time=0.537s

[87/600] breast_cancer   | linear_svm           | seed=149  | scaling_training_joint
 PASS | Macro-F1=0.893467 | BA=0.883929 | time=0.005s

[88/600] breast_cancer   | linear_svm           | seed=149  | sampling_training_joint
 PASS | Macro-F1=0.951576 | BA=0.940476 | time=0.006s

[89/600] breast_cancer   | logistic_regression  | seed=173  | scaling_training_joint
 PASS | Macro-F1=0.931367 | BA=0.916667 | time=0.006s

[90/600] breast_cancer   | logistic_regression  | seed=173  | sampling_training_joint
 PASS | Macro-F1=0.961911 | BA=0.957341 | time=0.007s
 CHECKPOINT SAVED

[91/600] breast_cancer   | decision_tree        | seed=173  | scaling_training_joint
 PASS | Macro-F1=0.883077 | BA=0.872024 | time=0.008s

[92/600] breast_cancer   | decision_tree        | seed=173  | sampling_training_joint
 PASS | Macro-F1=0.912837 | BA=0.902778 | time=0.009s

[93/600] breast_cancer   | random_forest        | seed=173  | scaling_training_joint
 PASS | Macro-F1=0.942230 | BA=0.933532 | time=0.528s

[94/600] breast_cancer   | random_forest        | seed=173  | sampling_training_joint
 PASS | Macro-F1=0.942230 | BA=0.933532 | time=0.509s

[95/600] breast_cancer   | linear_svm           | seed=173  | scaling_training_joint
 PASS | Macro-F1=0.931367 | BA=0.916667 | time=0.005s

[96/600] breast_cancer   | linear_svm           | seed=173  | sampling_training_joint
 PASS | Macro-F1=0.943452 | BA=0.943452 | time=0.005s

[97/600] breast_cancer   | logistic_regression  | seed=211  | scaling_training_joint
 PASS | Macro-F1=0.990528 | BA=0.988095 | time=0.006s

[98/600] breast_cancer   | logistic_regression  | seed=211  | sampling_training_joint
 PASS | Macro-F1=0.990621 | BA=0.993056 | time=0.006s

[99/600] breast_cancer   | decision_tree        | seed=211  | scaling_training_joint
 PASS | Macro-F1=0.905754 | BA=0.905754 | time=0.006s

[100/600] breast_cancer   | decision_tree        | seed=211  | sampling_training_joint
 PASS | Macro-F1=0.961911 | BA=0.957341 | time=0.010s
 CHECKPOINT SAVED

[101/600] breast_cancer   | random_forest        | seed=211  | scaling_training_joint
 PASS | Macro-F1=0.990528 | BA=0.988095 | time=0.488s

[102/600] breast_cancer   | random_forest        | seed=211  | sampling_training_joint
 PASS | Macro-F1=0.981151 | BA=0.981151 | time=0.526s

[103/600] breast_cancer   | linear_svm           | seed=211  | scaling_training_joint
 PASS | Macro-F1=0.980956 | BA=0.976190 | time=0.013s

[104/600] breast_cancer   | linear_svm           | seed=211  | sampling_training_joint
 PASS | Macro-F1=0.971863 | BA=0.974206 | time=0.006s

[105/600] breast_cancer   | logistic_regression  | seed=257  | scaling_training_joint
 PASS | Macro-F1=0.971583 | BA=0.969246 | time=0.005s

[106/600] breast_cancer   | logistic_regression  | seed=257  | sampling_training_joint
 PASS | Macro-F1=0.971583 | BA=0.969246 | time=0.007s

[107/600] breast_cancer   | decision_tree        | seed=257  | scaling_training_joint
 PASS | Macro-F1=0.887979 | BA=0.891865 | time=0.008s

[108/600] breast_cancer   | decision_tree        | seed=257  | sampling_training_joint
 PASS | Macro-F1=0.906649 | BA=0.910714 | time=0.009s

[109/600] breast_cancer   | random_forest        | seed=257  | scaling_training_joint
 PASS | Macro-F1=0.934348 | BA=0.936508 | time=0.502s

[110/600] breast_cancer   | random_forest        | seed=257  | sampling_training_joint
 PASS | Macro-F1=0.925319 | BA=0.929563 | time=0.512s
 CHECKPOINT SAVED

[111/600] breast_cancer   | linear_svm           | seed=257  | scaling_training_joint
 PASS | Macro-F1=0.971583 | BA=0.969246 | time=0.005s

[112/600] breast_cancer   | linear_svm           | seed=257  | sampling_training_joint
 PASS | Macro-F1=0.962302 | BA=0.962302 | time=0.006s

[113/600] breast_cancer   | logistic_regression  | seed=311  | scaling_training_joint
 PASS | Macro-F1=0.961911 | BA=0.957341 | time=0.007s

[114/600] breast_cancer   | logistic_regression  | seed=311  | sampling_training_joint
 PASS | Macro-F1=0.981151 | BA=0.981151 | time=0.008s

[115/600] breast_cancer   | decision_tree        | seed=311  | scaling_training_joint
 PASS | Macro-F1=0.917064 | BA=0.927579 | time=0.007s

[116/600] breast_cancer   | decision_tree        | seed=311  | sampling_training_joint
 PASS | Macro-F1=0.908213 | BA=0.920635 | time=0.009s

[117/600] breast_cancer   | random_forest        | seed=311  | scaling_training_joint
 PASS | Macro-F1=0.952638 | BA=0.950397 | time=0.467s

[118/600] breast_cancer   | random_forest        | seed=311  | sampling_training_joint
 PASS | Macro-F1=0.953106 | BA=0.955357 | time=0.506s

[119/600] breast_cancer   | linear_svm           | seed=311  | scaling_training_joint
 PASS | Macro-F1=0.961911 | BA=0.957341 | time=0.005s

[120/600] breast_cancer   | linear_svm           | seed=311  | sampling_training_joint
 PASS | Macro-F1=0.971583 | BA=0.969246 | time=0.006s
 CHECKPOINT SAVED

[121/600] bank_marketing  | logistic_regression  | seed=42   | scaling_training_joint
 PASS | Macro-F1=0.702923 | BA=0.665615 | time=0.421s

[122/600] bank_marketing  | logistic_regression  | seed=42   | sampling_training_joint
 PASS | Macro-F1=0.730496 | BA=0.831975 | time=0.372s

[123/600] bank_marketing  | decision_tree        | seed=42   | scaling_training_joint
 PASS | Macro-F1=0.696605 | BA=0.697170 | time=0.387s

[124/600] bank_marketing  | decision_tree        | seed=42   | sampling_training_joint
 PASS | Macro-F1=0.688342 | BA=0.683134 | time=0.484s

[125/600] bank_marketing  | random_forest        | seed=42   | scaling_training_joint
 PASS | Macro-F1=0.717891 | BA=0.680202 | time=5.511s

[126/600] bank_marketing  | random_forest        | seed=42   | sampling_training_joint
 PASS | Macro-F1=0.748254 | BA=0.730572 | time=6.733s

[127/600] bank_marketing  | linear_svm           | seed=42   | scaling_training_joint
 PASS | Macro-F1=0.623442 | BA=0.592231 | time=0.294s

[128/600] bank_marketing  | linear_svm           | seed=42   | sampling_training_joint
 PASS | Macro-F1=0.736488 | BA=0.831507 | time=0.242s

[129/600] bank_marketing  | logistic_regression  | seed=123  | scaling_training_joint
 PASS | Macro-F1=0.691224 | BA=0.651898 | time=0.448s

[130/600] bank_marketing  | logistic_regression  | seed=123  | sampling_training_joint
 PASS | Macro-F1=0.712503 | BA=0.813990 | time=0.338s
 CHECKPOINT SAVED

[131/600] bank_marketing  | decision_tree        | seed=123  | scaling_training_joint
 PASS | Macro-F1=0.701086 | BA=0.705813 | time=0.600s

[132/600] bank_marketing  | decision_tree        | seed=123  | sampling_training_joint
 PASS | Macro-F1=0.688993 | BA=0.684301 | time=0.615s

[133/600] bank_marketing  | random_forest        | seed=123  | scaling_training_joint
 PASS | Macro-F1=0.704735 | BA=0.665102 | time=3.277s

[134/600] bank_marketing  | random_forest        | seed=123  | sampling_training_joint
 PASS | Macro-F1=0.734445 | BA=0.713587 | time=4.294s

[135/600] bank_marketing  | linear_svm           | seed=123  | scaling_training_joint
 PASS | Macro-F1=0.665112 | BA=0.624589 | time=0.195s

[136/600] bank_marketing  | linear_svm           | seed=123  | sampling_training_joint
 PASS | Macro-F1=0.717694 | BA=0.814661 | time=0.205s

[137/600] bank_marketing  | logistic_regression  | seed=2026 | scaling_training_joint
 PASS | Macro-F1=0.706045 | BA=0.666742 | time=0.454s

[138/600] bank_marketing  | logistic_regression  | seed=2026 | sampling_training_joint
 PASS | Macro-F1=0.728829 | BA=0.837288 | time=0.235s

[139/600] bank_marketing  | decision_tree        | seed=2026 | scaling_training_joint
 PASS | Macro-F1=0.703792 | BA=0.708837 | time=0.308s

[140/600] bank_marketing  | decision_tree        | seed=2026 | sampling_training_joint
 PASS | Macro-F1=0.697100 | BA=0.697990 | time=0.346s
 CHECKPOINT SAVED

[141/600] bank_marketing  | random_forest        | seed=2026 | scaling_training_joint
 PASS | Macro-F1=0.726833 | BA=0.691015 | time=3.712s

[142/600] bank_marketing  | random_forest        | seed=2026 | sampling_training_joint
 PASS | Macro-F1=0.739344 | BA=0.723746 | time=4.310s

[143/600] bank_marketing  | linear_svm           | seed=2026 | scaling_training_joint
 PASS | Macro-F1=0.618178 | BA=0.588228 | time=0.166s

[144/600] bank_marketing  | linear_svm           | seed=2026 | sampling_training_joint
 PASS | Macro-F1=0.731339 | BA=0.835130 | time=0.194s

[145/600] bank_marketing  | logistic_regression  | seed=7    | scaling_training_joint
 PASS | Macro-F1=0.695543 | BA=0.654317 | time=0.248s

[146/600] bank_marketing  | logistic_regression  | seed=7    | sampling_training_joint
 PASS | Macro-F1=0.735058 | BA=0.839377 | time=0.231s

[147/600] bank_marketing  | decision_tree        | seed=7    | scaling_training_joint
 PASS | Macro-F1=0.710471 | BA=0.712634 | time=0.310s

[148/600] bank_marketing  | decision_tree        | seed=7    | sampling_training_joint
 PASS | Macro-F1=0.703935 | BA=0.702766 | time=0.340s

[149/600] bank_marketing  | random_forest        | seed=7    | scaling_training_joint
 PASS | Macro-F1=0.721720 | BA=0.682149 | time=2.974s

[150/600] bank_marketing  | random_forest        | seed=7    | sampling_training_joint
 PASS | Macro-F1=0.754057 | BA=0.733618 | time=5.198s
 CHECKPOINT SAVED

[151/600] bank_marketing  | linear_svm           | seed=7    | scaling_training_joint
 PASS | Macro-F1=0.663875 | BA=0.623866 | time=0.194s

[152/600] bank_marketing  | linear_svm           | seed=7    | sampling_training_joint
 PASS | Macro-F1=0.739935 | BA=0.836609 | time=0.193s

[153/600] bank_marketing  | logistic_regression  | seed=19   | scaling_training_joint
 PASS | Macro-F1=0.684538 | BA=0.645817 | time=0.264s

[154/600] bank_marketing  | logistic_regression  | seed=19   | sampling_training_joint
 PASS | Macro-F1=0.735929 | BA=0.839121 | time=0.230s

[155/600] bank_marketing  | decision_tree        | seed=19   | scaling_training_joint
 PASS | Macro-F1=0.702043 | BA=0.704703 | time=0.316s

[156/600] bank_marketing  | decision_tree        | seed=19   | sampling_training_joint
 PASS | Macro-F1=0.709260 | BA=0.708403 | time=0.354s

[157/600] bank_marketing  | random_forest        | seed=19   | scaling_training_joint
 PASS | Macro-F1=0.706870 | BA=0.667624 | time=3.423s

[158/600] bank_marketing  | random_forest        | seed=19   | sampling_training_joint
 PASS | Macro-F1=0.739890 | BA=0.717487 | time=5.619s

[159/600] bank_marketing  | linear_svm           | seed=19   | scaling_training_joint
 PASS | Macro-F1=0.668841 | BA=0.629292 | time=0.189s

[160/600] bank_marketing  | linear_svm           | seed=19   | sampling_training_joint
 PASS | Macro-F1=0.740555 | BA=0.835532 | time=0.195s
 CHECKPOINT SAVED

[161/600] bank_marketing  | logistic_regression  | seed=31   | scaling_training_joint
 PASS | Macro-F1=0.696920 | BA=0.659659 | time=0.273s

[162/600] bank_marketing  | logistic_regression  | seed=31   | sampling_training_joint
 PASS | Macro-F1=0.729169 | BA=0.836434 | time=0.226s

[163/600] bank_marketing  | decision_tree        | seed=31   | scaling_training_joint
 PASS | Macro-F1=0.698673 | BA=0.700797 | time=0.302s

[164/600] bank_marketing  | decision_tree        | seed=31   | sampling_training_joint
 PASS | Macro-F1=0.703641 | BA=0.703558 | time=0.351s

[165/600] bank_marketing  | random_forest        | seed=31   | scaling_training_joint
 PASS | Macro-F1=0.715627 | BA=0.676985 | time=2.955s

[166/600] bank_marketing  | random_forest        | seed=31   | sampling_training_joint
 PASS | Macro-F1=0.750213 | BA=0.732588 | time=4.400s

[167/600] bank_marketing  | linear_svm           | seed=31   | scaling_training_joint
 PASS | Macro-F1=0.664217 | BA=0.626241 | time=0.202s

[168/600] bank_marketing  | linear_svm           | seed=31   | sampling_training_joint
 PASS | Macro-F1=0.731465 | BA=0.835192 | time=0.229s

[169/600] bank_marketing  | logistic_regression  | seed=53   | scaling_training_joint
 PASS | Macro-F1=0.680462 | BA=0.643459 | time=0.898s

[170/600] bank_marketing  | logistic_regression  | seed=53   | sampling_training_joint
 PASS | Macro-F1=0.717893 | BA=0.822177 | time=0.318s
 CHECKPOINT SAVED

[171/600] bank_marketing  | decision_tree        | seed=53   | scaling_training_joint
 PASS | Macro-F1=0.683224 | BA=0.686699 | time=0.406s

[172/600] bank_marketing  | decision_tree        | seed=53   | sampling_training_joint
 PASS | Macro-F1=0.688092 | BA=0.693065 | time=0.349s

[173/600] bank_marketing  | random_forest        | seed=53   | scaling_training_joint
 PASS | Macro-F1=0.696896 | BA=0.661082 | time=2.890s

[174/600] bank_marketing  | random_forest        | seed=53   | sampling_training_joint
 PASS | Macro-F1=0.735987 | BA=0.718638 | time=4.238s

[175/600] bank_marketing  | linear_svm           | seed=53   | scaling_training_joint
 PASS | Macro-F1=0.671226 | BA=0.632259 | time=0.201s

[176/600] bank_marketing  | linear_svm           | seed=53   | sampling_training_joint
 PASS | Macro-F1=0.724211 | BA=0.822022 | time=0.193s

[177/600] bank_marketing  | logistic_regression  | seed=71   | scaling_training_joint
 PASS | Macro-F1=0.705764 | BA=0.665762 | time=0.260s

[178/600] bank_marketing  | logistic_regression  | seed=71   | sampling_training_joint
 PASS | Macro-F1=0.727787 | BA=0.822796 | time=0.232s

[179/600] bank_marketing  | decision_tree        | seed=71   | scaling_training_joint
 PASS | Macro-F1=0.695226 | BA=0.691168 | time=0.303s

[180/600] bank_marketing  | decision_tree        | seed=71   | sampling_training_joint
 PASS | Macro-F1=0.689592 | BA=0.687809 | time=0.360s
 CHECKPOINT SAVED

[181/600] bank_marketing  | random_forest        | seed=71   | scaling_training_joint
 PASS | Macro-F1=0.712065 | BA=0.676017 | time=3.906s

[182/600] bank_marketing  | random_forest        | seed=71   | sampling_training_joint
 PASS | Macro-F1=0.740833 | BA=0.722350 | time=5.069s

[183/600] bank_marketing  | linear_svm           | seed=71   | scaling_training_joint
 PASS | Macro-F1=0.664268 | BA=0.623769 | time=0.197s

[184/600] bank_marketing  | linear_svm           | seed=71   | sampling_training_joint
 PASS | Macro-F1=0.733426 | BA=0.823530 | time=0.206s

[185/600] bank_marketing  | logistic_regression  | seed=89   | scaling_training_joint
 PASS | Macro-F1=0.693982 | BA=0.653469 | time=0.527s

[186/600] bank_marketing  | logistic_regression  | seed=89   | sampling_training_joint
 PASS | Macro-F1=0.731476 | BA=0.839993 | time=0.218s

[187/600] bank_marketing  | decision_tree        | seed=89   | scaling_training_joint
 PASS | Macro-F1=0.691586 | BA=0.694346 | time=0.310s

[188/600] bank_marketing  | decision_tree        | seed=89   | sampling_training_joint
 PASS | Macro-F1=0.679293 | BA=0.678779 | time=0.347s

[189/600] bank_marketing  | random_forest        | seed=89   | scaling_training_joint
 PASS | Macro-F1=0.716590 | BA=0.674610 | time=3.760s

[190/600] bank_marketing  | random_forest        | seed=89   | sampling_training_joint
 PASS | Macro-F1=0.744178 | BA=0.722908 | time=4.558s
 CHECKPOINT SAVED

[191/600] bank_marketing  | linear_svm           | seed=89   | scaling_training_joint
 PASS | Macro-F1=0.668025 | BA=0.626952 | time=0.192s

[192/600] bank_marketing  | linear_svm           | seed=89   | sampling_training_joint
 PASS | Macro-F1=0.735532 | BA=0.838238 | time=0.185s

[193/600] bank_marketing  | logistic_regression  | seed=107  | scaling_training_joint
 PASS | Macro-F1=0.694392 | BA=0.658719 | time=0.411s

[194/600] bank_marketing  | logistic_regression  | seed=107  | sampling_training_joint
 PASS | Macro-F1=0.727550 | BA=0.831862 | time=0.235s

[195/600] bank_marketing  | decision_tree        | seed=107  | scaling_training_joint
 PASS | Macro-F1=0.722071 | BA=0.733952 | time=0.325s

[196/600] bank_marketing  | decision_tree        | seed=107  | sampling_training_joint
 PASS | Macro-F1=0.710335 | BA=0.709473 | time=0.377s

[197/600] bank_marketing  | random_forest        | seed=107  | scaling_training_joint
 PASS | Macro-F1=0.721119 | BA=0.683163 | time=3.024s

[198/600] bank_marketing  | random_forest        | seed=107  | sampling_training_joint
 PASS | Macro-F1=0.752534 | BA=0.736055 | time=5.259s

[199/600] bank_marketing  | linear_svm           | seed=107  | scaling_training_joint
 PASS | Macro-F1=0.666724 | BA=0.627465 | time=0.187s

[200/600] bank_marketing  | linear_svm           | seed=107  | sampling_training_joint
 PASS | Macro-F1=0.729279 | BA=0.825917 | time=0.198s
 CHECKPOINT SAVED

[201/600] bank_marketing  | logistic_regression  | seed=149  | scaling_training_joint
 PASS | Macro-F1=0.703900 | BA=0.664219 | time=0.259s

[202/600] bank_marketing  | logistic_regression  | seed=149  | sampling_training_joint
 PASS | Macro-F1=0.731171 | BA=0.837783 | time=0.225s

[203/600] bank_marketing  | decision_tree        | seed=149  | scaling_training_joint
 PASS | Macro-F1=0.693337 | BA=0.694528 | time=0.294s

[204/600] bank_marketing  | decision_tree        | seed=149  | sampling_training_joint
 PASS | Macro-F1=0.692600 | BA=0.691812 | time=0.360s

[205/600] bank_marketing  | random_forest        | seed=149  | scaling_training_joint
 PASS | Macro-F1=0.714908 | BA=0.677936 | time=2.985s

[206/600] bank_marketing  | random_forest        | seed=149  | sampling_training_joint
 PASS | Macro-F1=0.743677 | BA=0.724679 | time=5.216s

[207/600] bank_marketing  | linear_svm           | seed=149  | scaling_training_joint
 PASS | Macro-F1=0.637316 | BA=0.602183 | time=0.192s

[208/600] bank_marketing  | linear_svm           | seed=149  | sampling_training_joint
 PASS | Macro-F1=0.733080 | BA=0.835312 | time=0.191s

[209/600] bank_marketing  | logistic_regression  | seed=173  | scaling_training_joint
 PASS | Macro-F1=0.688621 | BA=0.649819 | time=0.251s

[210/600] bank_marketing  | logistic_regression  | seed=173  | sampling_training_joint
 PASS | Macro-F1=0.722756 | BA=0.824335 | time=0.227s
 CHECKPOINT SAVED

[211/600] bank_marketing  | decision_tree        | seed=173  | scaling_training_joint
 PASS | Macro-F1=0.703178 | BA=0.707954 | time=0.317s

[212/600] bank_marketing  | decision_tree        | seed=173  | sampling_training_joint
 PASS | Macro-F1=0.692608 | BA=0.690325 | time=0.373s

[213/600] bank_marketing  | random_forest        | seed=173  | scaling_training_joint
 PASS | Macro-F1=0.711528 | BA=0.674628 | time=2.974s

[214/600] bank_marketing  | random_forest        | seed=173  | sampling_training_joint
 PASS | Macro-F1=0.751484 | BA=0.735680 | time=4.329s

[215/600] bank_marketing  | linear_svm           | seed=173  | scaling_training_joint
 PASS | Macro-F1=0.644482 | BA=0.607951 | time=0.183s

[216/600] bank_marketing  | linear_svm           | seed=173  | sampling_training_joint
 PASS | Macro-F1=0.728859 | BA=0.821970 | time=0.197s

[217/600] bank_marketing  | logistic_regression  | seed=211  | scaling_training_joint
 PASS | Macro-F1=0.689412 | BA=0.652063 | time=0.358s

[218/600] bank_marketing  | logistic_regression  | seed=211  | sampling_training_joint
 PASS | Macro-F1=0.727831 | BA=0.829618 | time=0.315s

[219/600] bank_marketing  | decision_tree        | seed=211  | scaling_training_joint
 PASS | Macro-F1=0.707531 | BA=0.714479 | time=0.410s

[220/600] bank_marketing  | decision_tree        | seed=211  | sampling_training_joint
 PASS | Macro-F1=0.694872 | BA=0.691612 | time=0.473s
 CHECKPOINT SAVED

[221/600] bank_marketing  | random_forest        | seed=211  | scaling_training_joint
 PASS | Macro-F1=0.710099 | BA=0.669953 | time=3.065s

[222/600] bank_marketing  | random_forest        | seed=211  | sampling_training_joint
 PASS | Macro-F1=0.741909 | BA=0.719503 | time=4.363s

[223/600] bank_marketing  | linear_svm           | seed=211  | scaling_training_joint
 PASS | Macro-F1=0.670667 | BA=0.631787 | time=0.195s

[224/600] bank_marketing  | linear_svm           | seed=211  | sampling_training_joint
 PASS | Macro-F1=0.730655 | BA=0.826258 | time=0.189s

[225/600] bank_marketing  | logistic_regression  | seed=257  | scaling_training_joint
 PASS | Macro-F1=0.714654 | BA=0.676040 | time=0.264s

[226/600] bank_marketing  | logistic_regression  | seed=257  | sampling_training_joint
 PASS | Macro-F1=0.729355 | BA=0.838581 | time=0.233s

[227/600] bank_marketing  | decision_tree        | seed=257  | scaling_training_joint
 PASS | Macro-F1=0.702082 | BA=0.707168 | time=0.304s

[228/600] bank_marketing  | decision_tree        | seed=257  | sampling_training_joint
 PASS | Macro-F1=0.703534 | BA=0.705961 | time=0.363s

[229/600] bank_marketing  | random_forest        | seed=257  | scaling_training_joint
 PASS | Macro-F1=0.731140 | BA=0.694067 | time=3.909s

[230/600] bank_marketing  | random_forest        | seed=257  | sampling_training_joint
 PASS | Macro-F1=0.755396 | BA=0.740058 | time=4.441s
 CHECKPOINT SAVED

[231/600] bank_marketing  | linear_svm           | seed=257  | scaling_training_joint
 PASS | Macro-F1=0.624578 | BA=0.592350 | time=0.176s

[232/600] bank_marketing  | linear_svm           | seed=257  | sampling_training_joint
 PASS | Macro-F1=0.736572 | BA=0.839781 | time=0.195s

[233/600] bank_marketing  | logistic_regression  | seed=311  | scaling_training_joint
 PASS | Macro-F1=0.686387 | BA=0.650366 | time=0.293s

[234/600] bank_marketing  | logistic_regression  | seed=311  | sampling_training_joint
 PASS | Macro-F1=0.728415 | BA=0.828889 | time=0.223s

[235/600] bank_marketing  | decision_tree        | seed=311  | scaling_training_joint
 PASS | Macro-F1=0.705279 | BA=0.706712 | time=0.312s

[236/600] bank_marketing  | decision_tree        | seed=311  | sampling_training_joint
 PASS | Macro-F1=0.702422 | BA=0.697556 | time=0.373s

[237/600] bank_marketing  | random_forest        | seed=311  | scaling_training_joint
 PASS | Macro-F1=0.709794 | BA=0.672516 | time=2.862s

[238/600] bank_marketing  | random_forest        | seed=311  | sampling_training_joint
 PASS | Macro-F1=0.743478 | BA=0.722657 | time=5.263s

[239/600] bank_marketing  | linear_svm           | seed=311  | scaling_training_joint
 PASS | Macro-F1=0.613839 | BA=0.585204 | time=0.181s

[240/600] bank_marketing  | linear_svm           | seed=311  | sampling_training_joint
 PASS | Macro-F1=0.732066 | BA=0.827294 | time=0.192s
 CHECKPOINT SAVED

[241/600] pima_diabetes   | logistic_regression  | seed=42   | scaling_training_joint
 PASS | Macro-F1=0.665072 | BA=0.660000 | time=0.005s

[242/600] pima_diabetes   | logistic_regression  | seed=42   | sampling_training_joint
 PASS | Macro-F1=0.732960 | BA=0.745370 | time=0.005s

[243/600] pima_diabetes   | decision_tree        | seed=42   | scaling_training_joint
 PASS | Macro-F1=0.618182 | BA=0.619259 | time=0.004s

[244/600] pima_diabetes   | decision_tree        | seed=42   | sampling_training_joint
 PASS | Macro-F1=0.718654 | BA=0.712037 | time=0.006s

[245/600] pima_diabetes   | random_forest        | seed=42   | scaling_training_joint
 PASS | Macro-F1=0.709729 | BA=0.702778 | time=0.359s

[246/600] pima_diabetes   | random_forest        | seed=42   | sampling_training_joint
 PASS | Macro-F1=0.718227 | BA=0.715556 | time=0.362s

[247/600] pima_diabetes   | linear_svm           | seed=42   | scaling_training_joint
 PASS | Macro-F1=0.670748 | BA=0.665000 | time=0.003s

[248/600] pima_diabetes   | linear_svm           | seed=42   | sampling_training_joint
 PASS | Macro-F1=0.732960 | BA=0.745370 | time=0.003s

[249/600] pima_diabetes   | logistic_regression  | seed=123  | scaling_training_joint
 PASS | Macro-F1=0.707660 | BA=0.695741 | time=0.004s

[250/600] pima_diabetes   | logistic_regression  | seed=123  | sampling_training_joint
 PASS | Macro-F1=0.745455 | BA=0.747593 | time=0.003s
 CHECKPOINT SAVED

[251/600] pima_diabetes   | decision_tree        | seed=123  | scaling_training_joint
 PASS | Macro-F1=0.642624 | BA=0.640000 | time=0.004s

[252/600] pima_diabetes   | decision_tree        | seed=123  | sampling_training_joint
 PASS | Macro-F1=0.646358 | BA=0.641481 | time=0.005s

[253/600] pima_diabetes   | random_forest        | seed=123  | scaling_training_joint
 PASS | Macro-F1=0.718409 | BA=0.708519 | time=0.350s

[254/600] pima_diabetes   | random_forest        | seed=123  | sampling_training_joint
 PASS | Macro-F1=0.738688 | BA=0.734815 | time=0.367s

[255/600] pima_diabetes   | linear_svm           | seed=123  | scaling_training_joint
 PASS | Macro-F1=0.721014 | BA=0.709259 | time=0.004s

[256/600] pima_diabetes   | linear_svm           | seed=123  | sampling_training_joint
 PASS | Macro-F1=0.743333 | BA=0.743333 | time=0.004s

[257/600] pima_diabetes   | logistic_regression  | seed=2026 | scaling_training_joint
 PASS | Macro-F1=0.638940 | BA=0.633704 | time=0.003s

[258/600] pima_diabetes   | logistic_regression  | seed=2026 | sampling_training_joint
 PASS | Macro-F1=0.660606 | BA=0.662037 | time=0.004s

[259/600] pima_diabetes   | decision_tree        | seed=2026 | scaling_training_joint
 PASS | Macro-F1=0.580141 | BA=0.578704 | time=0.004s

[260/600] pima_diabetes   | decision_tree        | seed=2026 | sampling_training_joint
 PASS | Macro-F1=0.663458 | BA=0.662778 | time=0.005s
 CHECKPOINT SAVED

[261/600] pima_diabetes   | random_forest        | seed=2026 | scaling_training_joint
 PASS | Macro-F1=0.640816 | BA=0.636481 | time=0.347s

[262/600] pima_diabetes   | random_forest        | seed=2026 | sampling_training_joint
 PASS | Macro-F1=0.651584 | BA=0.649259 | time=0.357s

[263/600] pima_diabetes   | linear_svm           | seed=2026 | scaling_training_joint
 PASS | Macro-F1=0.623242 | BA=0.619444 | time=0.003s

[264/600] pima_diabetes   | linear_svm           | seed=2026 | sampling_training_joint
 PASS | Macro-F1=0.657778 | BA=0.657778 | time=0.004s

[265/600] pima_diabetes   | logistic_regression  | seed=7    | scaling_training_joint
 PASS | Macro-F1=0.607531 | BA=0.605926 | time=0.003s

[266/600] pima_diabetes   | logistic_regression  | seed=7    | sampling_training_joint
 PASS | Macro-F1=0.717456 | BA=0.726852 | time=0.004s

[267/600] pima_diabetes   | decision_tree        | seed=7    | scaling_training_joint
 PASS | Macro-F1=0.646358 | BA=0.641481 | time=0.004s

[268/600] pima_diabetes   | decision_tree        | seed=7    | sampling_training_joint
 PASS | Macro-F1=0.659423 | BA=0.655000 | time=0.005s

[269/600] pima_diabetes   | random_forest        | seed=7    | scaling_training_joint
 PASS | Macro-F1=0.715029 | BA=0.704259 | time=0.358s

[270/600] pima_diabetes   | random_forest        | seed=7    | sampling_training_joint
 PASS | Macro-F1=0.697327 | BA=0.689259 | time=0.359s
 CHECKPOINT SAVED

[271/600] pima_diabetes   | linear_svm           | seed=7    | scaling_training_joint
 PASS | Macro-F1=0.736649 | BA=0.727037 | time=0.003s

[272/600] pima_diabetes   | linear_svm           | seed=7    | sampling_training_joint
 PASS | Macro-F1=0.717456 | BA=0.726852 | time=0.003s

[273/600] pima_diabetes   | logistic_regression  | seed=19   | scaling_training_joint
 PASS | Macro-F1=0.703188 | BA=0.694259 | time=0.004s

[274/600] pima_diabetes   | logistic_regression  | seed=19   | sampling_training_joint
 PASS | Macro-F1=0.740603 | BA=0.754630 | time=0.004s

[275/600] pima_diabetes   | decision_tree        | seed=19   | scaling_training_joint
 PASS | Macro-F1=0.668103 | BA=0.674815 | time=0.003s

[276/600] pima_diabetes   | decision_tree        | seed=19   | sampling_training_joint
 PASS | Macro-F1=0.646465 | BA=0.647778 | time=0.005s

[277/600] pima_diabetes   | random_forest        | seed=19   | scaling_training_joint
 PASS | Macro-F1=0.720742 | BA=0.719815 | time=0.347s

[278/600] pima_diabetes   | random_forest        | seed=19   | sampling_training_joint
 PASS | Macro-F1=0.717172 | BA=0.719074 | time=0.350s

[279/600] pima_diabetes   | linear_svm           | seed=19   | scaling_training_joint
 PASS | Macro-F1=0.724615 | BA=0.717037 | time=0.004s

[280/600] pima_diabetes   | linear_svm           | seed=19   | sampling_training_joint
 PASS | Macro-F1=0.726950 | BA=0.740370 | time=0.004s
 CHECKPOINT SAVED

[281/600] pima_diabetes   | logistic_regression  | seed=31   | scaling_training_joint
 PASS | Macro-F1=0.711450 | BA=0.700000 | time=0.004s

[282/600] pima_diabetes   | logistic_regression  | seed=31   | sampling_training_joint
 PASS | Macro-F1=0.696174 | BA=0.727407 | time=0.004s

[283/600] pima_diabetes   | decision_tree        | seed=31   | scaling_training_joint
 PASS | Macro-F1=0.632260 | BA=0.637037 | time=0.004s

[284/600] pima_diabetes   | decision_tree        | seed=31   | sampling_training_joint
 PASS | Macro-F1=0.581117 | BA=0.581481 | time=0.005s

[285/600] pima_diabetes   | random_forest        | seed=31   | scaling_training_joint
 PASS | Macro-F1=0.706421 | BA=0.705556 | time=0.350s

[286/600] pima_diabetes   | random_forest        | seed=31   | sampling_training_joint
 PASS | Macro-F1=0.674246 | BA=0.687593 | time=0.354s

[287/600] pima_diabetes   | linear_svm           | seed=31   | scaling_training_joint
 PASS | Macro-F1=0.715558 | BA=0.711296 | time=0.005s

[288/600] pima_diabetes   | linear_svm           | seed=31   | sampling_training_joint
 PASS | Macro-F1=0.681818 | BA=0.708889 | time=0.004s

[289/600] pima_diabetes   | logistic_regression  | seed=53   | scaling_training_joint
 PASS | Macro-F1=0.661219 | BA=0.653704 | time=0.004s

[290/600] pima_diabetes   | logistic_regression  | seed=53   | sampling_training_joint
 PASS | Macro-F1=0.705357 | BA=0.709074 | time=0.004s
 CHECKPOINT SAVED

[291/600] pima_diabetes   | decision_tree        | seed=53   | scaling_training_joint
 PASS | Macro-F1=0.640445 | BA=0.646296 | time=0.004s

[292/600] pima_diabetes   | decision_tree        | seed=53   | sampling_training_joint
 PASS | Macro-F1=0.626743 | BA=0.628519 | time=0.005s

[293/600] pima_diabetes   | random_forest        | seed=53   | scaling_training_joint
 PASS | Macro-F1=0.706834 | BA=0.702037 | time=0.357s

[294/600] pima_diabetes   | random_forest        | seed=53   | sampling_training_joint
 PASS | Macro-F1=0.688889 | BA=0.690556 | time=0.346s

[295/600] pima_diabetes   | linear_svm           | seed=53   | scaling_training_joint
 PASS | Macro-F1=0.677352 | BA=0.667963 | time=0.003s

[296/600] pima_diabetes   | linear_svm           | seed=53   | sampling_training_joint
 PASS | Macro-F1=0.717172 | BA=0.719074 | time=0.004s

[297/600] pima_diabetes   | logistic_regression  | seed=71   | scaling_training_joint
 PASS | Macro-F1=0.766746 | BA=0.755556 | time=0.004s

[298/600] pima_diabetes   | logistic_regression  | seed=71   | sampling_training_joint
 PASS | Macro-F1=0.763248 | BA=0.770370 | time=0.004s

[299/600] pima_diabetes   | decision_tree        | seed=71   | scaling_training_joint
 PASS | Macro-F1=0.640816 | BA=0.636481 | time=0.004s

[300/600] pima_diabetes   | decision_tree        | seed=71   | sampling_training_joint
 PASS | Macro-F1=0.730144 | BA=0.725556 | time=0.006s
 CHECKPOINT SAVED

[301/600] pima_diabetes   | random_forest        | seed=71   | scaling_training_joint
 PASS | Macro-F1=0.682193 | BA=0.675000 | time=0.341s

[302/600] pima_diabetes   | random_forest        | seed=71   | sampling_training_joint
 PASS | Macro-F1=0.700971 | BA=0.697037 | time=0.350s

[303/600] pima_diabetes   | linear_svm           | seed=71   | scaling_training_joint
 PASS | Macro-F1=0.760544 | BA=0.750556 | time=0.003s

[304/600] pima_diabetes   | linear_svm           | seed=71   | sampling_training_joint
 PASS | Macro-F1=0.763248 | BA=0.770370 | time=0.004s

[305/600] pima_diabetes   | logistic_regression  | seed=89   | scaling_training_joint
 PASS | Macro-F1=0.684221 | BA=0.675741 | time=0.005s

[306/600] pima_diabetes   | logistic_regression  | seed=89   | sampling_training_joint
 PASS | Macro-F1=0.693615 | BA=0.699074 | time=0.004s

[307/600] pima_diabetes   | decision_tree        | seed=89   | scaling_training_joint
 PASS | Macro-F1=0.657211 | BA=0.654259 | time=0.004s

[308/600] pima_diabetes   | decision_tree        | seed=89   | sampling_training_joint
 PASS | Macro-F1=0.667060 | BA=0.660741 | time=0.005s

[309/600] pima_diabetes   | random_forest        | seed=89   | scaling_training_joint
 PASS | Macro-F1=0.686384 | BA=0.682778 | time=0.354s

[310/600] pima_diabetes   | random_forest        | seed=89   | sampling_training_joint
 PASS | Macro-F1=0.697926 | BA=0.696296 | time=0.348s
 CHECKPOINT SAVED

[311/600] pima_diabetes   | linear_svm           | seed=89   | scaling_training_joint
 PASS | Macro-F1=0.684221 | BA=0.675741 | time=0.003s

[312/600] pima_diabetes   | linear_svm           | seed=89   | sampling_training_joint
 PASS | Macro-F1=0.699478 | BA=0.704074 | time=0.007s

[313/600] pima_diabetes   | logistic_regression  | seed=107  | scaling_training_joint
 PASS | Macro-F1=0.745578 | BA=0.736296 | time=0.006s

[314/600] pima_diabetes   | logistic_regression  | seed=107  | sampling_training_joint
 PASS | Macro-F1=0.738722 | BA=0.762407 | time=0.006s

[315/600] pima_diabetes   | decision_tree        | seed=107  | scaling_training_joint
 PASS | Macro-F1=0.608520 | BA=0.620556 | time=0.007s

[316/600] pima_diabetes   | decision_tree        | seed=107  | sampling_training_joint
 PASS | Macro-F1=0.673206 | BA=0.695370 | time=0.008s

[317/600] pima_diabetes   | random_forest        | seed=107  | scaling_training_joint
 PASS | Macro-F1=0.680618 | BA=0.677778 | time=0.422s

[318/600] pima_diabetes   | random_forest        | seed=107  | sampling_training_joint
 PASS | Macro-F1=0.711508 | BA=0.721852 | time=0.423s

[319/600] pima_diabetes   | linear_svm           | seed=107  | scaling_training_joint
 PASS | Macro-F1=0.727460 | BA=0.721296 | time=0.004s

[320/600] pima_diabetes   | linear_svm           | seed=107  | sampling_training_joint
 PASS | Macro-F1=0.725323 | BA=0.748148 | time=0.007s
 CHECKPOINT SAVED

[321/600] pima_diabetes   | logistic_regression  | seed=149  | scaling_training_joint
 PASS | Macro-F1=0.690016 | BA=0.680741 | time=0.006s

[322/600] pima_diabetes   | logistic_regression  | seed=149  | sampling_training_joint
 PASS | Macro-F1=0.763248 | BA=0.770370 | time=0.006s

[323/600] pima_diabetes   | decision_tree        | seed=149  | scaling_training_joint
 PASS | Macro-F1=0.695452 | BA=0.711111 | time=0.006s

[324/600] pima_diabetes   | decision_tree        | seed=149  | sampling_training_joint
 PASS | Macro-F1=0.699478 | BA=0.704074 | time=0.007s

[325/600] pima_diabetes   | random_forest        | seed=149  | scaling_training_joint
 PASS | Macro-F1=0.706552 | BA=0.698519 | time=0.459s

[326/600] pima_diabetes   | random_forest        | seed=149  | sampling_training_joint
 PASS | Macro-F1=0.727460 | BA=0.721296 | time=0.439s

[327/600] pima_diabetes   | linear_svm           | seed=149  | scaling_training_joint
 PASS | Macro-F1=0.733630 | BA=0.722778 | time=0.003s

[328/600] pima_diabetes   | linear_svm           | seed=149  | sampling_training_joint
 PASS | Macro-F1=0.761480 | BA=0.766111 | time=0.004s

[329/600] pima_diabetes   | logistic_regression  | seed=173  | scaling_training_joint
 PASS | Macro-F1=0.691503 | BA=0.684259 | time=0.003s

[330/600] pima_diabetes   | logistic_regression  | seed=173  | sampling_training_joint
 PASS | Macro-F1=0.704518 | BA=0.724630 | time=0.004s
 CHECKPOINT SAVED

[331/600] pima_diabetes   | decision_tree        | seed=173  | scaling_training_joint
 PASS | Macro-F1=0.577533 | BA=0.577222 | time=0.004s

[332/600] pima_diabetes   | decision_tree        | seed=173  | sampling_training_joint
 PASS | Macro-F1=0.625850 | BA=0.622222 | time=0.005s

[333/600] pima_diabetes   | random_forest        | seed=173  | scaling_training_joint
 PASS | Macro-F1=0.689327 | BA=0.687037 | time=0.345s

[334/600] pima_diabetes   | random_forest        | seed=173  | sampling_training_joint
 PASS | Macro-F1=0.692100 | BA=0.691296 | time=0.364s

[335/600] pima_diabetes   | linear_svm           | seed=173  | scaling_training_joint
 PASS | Macro-F1=0.695136 | BA=0.692037 | time=0.003s

[336/600] pima_diabetes   | linear_svm           | seed=173  | sampling_training_joint
 PASS | Macro-F1=0.708987 | BA=0.725370 | time=0.003s

[337/600] pima_diabetes   | logistic_regression  | seed=211  | scaling_training_joint
 PASS | Macro-F1=0.691503 | BA=0.684259 | time=0.003s

[338/600] pima_diabetes   | logistic_regression  | seed=211  | sampling_training_joint
 PASS | Macro-F1=0.720953 | BA=0.735370 | time=0.004s

[339/600] pima_diabetes   | decision_tree        | seed=211  | scaling_training_joint
 PASS | Macro-F1=0.600741 | BA=0.600741 | time=0.003s

[340/600] pima_diabetes   | decision_tree        | seed=211  | sampling_training_joint
 PASS | Macro-F1=0.548891 | BA=0.548704 | time=0.005s
 CHECKPOINT SAVED

[341/600] pima_diabetes   | random_forest        | seed=211  | scaling_training_joint
 PASS | Macro-F1=0.620415 | BA=0.617222 | time=0.350s

[342/600] pima_diabetes   | random_forest        | seed=211  | sampling_training_joint
 PASS | Macro-F1=0.637914 | BA=0.638519 | time=0.357s

[343/600] pima_diabetes   | linear_svm           | seed=211  | scaling_training_joint
 PASS | Macro-F1=0.697327 | BA=0.689259 | time=0.003s

[344/600] pima_diabetes   | linear_svm           | seed=211  | sampling_training_joint
 PASS | Macro-F1=0.713298 | BA=0.726111 | time=0.003s

[345/600] pima_diabetes   | logistic_regression  | seed=257  | scaling_training_joint
 PASS | Macro-F1=0.694843 | BA=0.688519 | time=0.003s

[346/600] pima_diabetes   | logistic_regression  | seed=257  | sampling_training_joint
 PASS | Macro-F1=0.743513 | BA=0.763148 | time=0.004s

[347/600] pima_diabetes   | decision_tree        | seed=257  | scaling_training_joint
 PASS | Macro-F1=0.739425 | BA=0.742593 | time=0.004s

[348/600] pima_diabetes   | decision_tree        | seed=257  | sampling_training_joint
 PASS | Macro-F1=0.657778 | BA=0.657778 | time=0.005s

[349/600] pima_diabetes   | random_forest        | seed=257  | scaling_training_joint
 PASS | Macro-F1=0.706421 | BA=0.705556 | time=0.343s

[350/600] pima_diabetes   | random_forest        | seed=257  | sampling_training_joint
 PASS | Macro-F1=0.753510 | BA=0.756852 | time=0.353s
 CHECKPOINT SAVED

[351/600] pima_diabetes   | linear_svm           | seed=257  | scaling_training_joint
 PASS | Macro-F1=0.748269 | BA=0.740556 | time=0.003s

[352/600] pima_diabetes   | linear_svm           | seed=257  | sampling_training_joint
 PASS | Macro-F1=0.740603 | BA=0.754630 | time=0.004s

[353/600] pima_diabetes   | logistic_regression  | seed=311  | scaling_training_joint
 PASS | Macro-F1=0.727047 | BA=0.714259 | time=0.004s

[354/600] pima_diabetes   | logistic_regression  | seed=311  | sampling_training_joint
 PASS | Macro-F1=0.798618 | BA=0.796111 | time=0.004s

[355/600] pima_diabetes   | decision_tree        | seed=311  | scaling_training_joint
 PASS | Macro-F1=0.618182 | BA=0.615185 | time=0.004s

[356/600] pima_diabetes   | decision_tree        | seed=311  | sampling_training_joint
 PASS | Macro-F1=0.616260 | BA=0.612963 | time=0.005s

[357/600] pima_diabetes   | random_forest        | seed=311  | scaling_training_joint
 PASS | Macro-F1=0.713636 | BA=0.700741 | time=0.356s

[358/600] pima_diabetes   | random_forest        | seed=311  | sampling_training_joint
 PASS | Macro-F1=0.736649 | BA=0.727037 | time=0.363s

[359/600] pima_diabetes   | linear_svm           | seed=311  | scaling_training_joint
 PASS | Macro-F1=0.705515 | BA=0.695000 | time=0.004s

[360/600] pima_diabetes   | linear_svm           | seed=311  | sampling_training_joint
 PASS | Macro-F1=0.792347 | BA=0.791111 | time=0.004s
 CHECKPOINT SAVED

[361/600] spambase        | logistic_regression  | seed=42   | scaling_training_joint
 PASS | Macro-F1=0.916649 | BA=0.914852 | time=0.018s

[362/600] spambase        | logistic_regression  | seed=42   | sampling_training_joint
 PASS | Macro-F1=0.905962 | BA=0.907336 | time=0.029s

[363/600] spambase        | decision_tree        | seed=42   | scaling_training_joint
 PASS | Macro-F1=0.879724 | BA=0.884386 | time=0.034s

[364/600] spambase        | decision_tree        | seed=42   | sampling_training_joint
 PASS | Macro-F1=0.889998 | BA=0.890940 | time=0.035s

[365/600] spambase        | random_forest        | seed=42   | scaling_training_joint
 PASS | Macro-F1=0.936027 | BA=0.933936 | time=0.578s

[366/600] spambase        | random_forest        | seed=42   | sampling_training_joint
 PASS | Macro-F1=0.939606 | BA=0.938550 | time=0.618s

[367/600] spambase        | linear_svm           | seed=42   | scaling_training_joint
 PASS | Macro-F1=0.909706 | BA=0.907551 | time=0.013s

[368/600] spambase        | linear_svm           | seed=42   | sampling_training_joint
 PASS | Macro-F1=0.905786 | BA=0.906373 | time=0.023s

[369/600] spambase        | logistic_regression  | seed=123  | scaling_training_joint
 PASS | Macro-F1=0.914011 | BA=0.910653 | time=0.016s

[370/600] spambase        | logistic_regression  | seed=123  | sampling_training_joint
 PASS | Macro-F1=0.914867 | BA=0.915467 | time=0.028s
 CHECKPOINT SAVED

[371/600] spambase        | decision_tree        | seed=123  | scaling_training_joint
 PASS | Macro-F1=0.896705 | BA=0.897279 | time=0.037s

[372/600] spambase        | decision_tree        | seed=123  | sampling_training_joint
 PASS | Macro-F1=0.889357 | BA=0.888052 | time=0.044s

[373/600] spambase        | random_forest        | seed=123  | scaling_training_joint
 PASS | Macro-F1=0.939298 | BA=0.936143 | time=0.605s

[374/600] spambase        | random_forest        | seed=123  | sampling_training_joint
 PASS | Macro-F1=0.938249 | BA=0.935728 | time=0.608s

[375/600] spambase        | linear_svm           | seed=123  | scaling_training_joint
 PASS | Macro-F1=0.910428 | BA=0.906521 | time=0.014s

[376/600] spambase        | linear_svm           | seed=123  | sampling_training_joint
 PASS | Macro-F1=0.917137 | BA=0.917740 | time=0.081s

[377/600] spambase        | logistic_regression  | seed=2026 | scaling_training_joint
 PASS | Macro-F1=0.912082 | BA=0.910305 | time=0.016s

[378/600] spambase        | logistic_regression  | seed=2026 | sampling_training_joint
 PASS | Macro-F1=0.912760 | BA=0.914156 | time=0.028s

[379/600] spambase        | decision_tree        | seed=2026 | scaling_training_joint
 PASS | Macro-F1=0.889417 | BA=0.892451 | time=0.040s

[380/600] spambase        | decision_tree        | seed=2026 | sampling_training_joint
 PASS | Macro-F1=0.889019 | BA=0.890525 | time=0.048s
 CHECKPOINT SAVED

[381/600] spambase        | random_forest        | seed=2026 | scaling_training_joint
 PASS | Macro-F1=0.952164 | BA=0.951294 | time=0.567s

[382/600] spambase        | random_forest        | seed=2026 | sampling_training_joint
 PASS | Macro-F1=0.950126 | BA=0.951428 | time=0.630s

[383/600] spambase        | linear_svm           | seed=2026 | scaling_training_joint
 PASS | Macro-F1=0.900563 | BA=0.898457 | time=0.015s

[384/600] spambase        | linear_svm           | seed=2026 | sampling_training_joint
 PASS | Macro-F1=0.913773 | BA=0.914571 | time=0.026s

[385/600] spambase        | logistic_regression  | seed=7    | scaling_training_joint
 PASS | Macro-F1=0.919785 | BA=0.916578 | time=0.022s

[386/600] spambase        | logistic_regression  | seed=7    | sampling_training_joint
 PASS | Macro-F1=0.922993 | BA=0.924628 | time=0.029s

[387/600] spambase        | decision_tree        | seed=7    | scaling_training_joint
 PASS | Macro-F1=0.886105 | BA=0.885364 | time=0.042s

[388/600] spambase        | decision_tree        | seed=7    | sampling_training_joint
 PASS | Macro-F1=0.881665 | BA=0.881298 | time=0.043s

[389/600] spambase        | random_forest        | seed=7    | scaling_training_joint
 PASS | Macro-F1=0.947232 | BA=0.943378 | time=0.593s

[390/600] spambase        | random_forest        | seed=7    | sampling_training_joint
 PASS | Macro-F1=0.936941 | BA=0.933388 | time=0.648s
 CHECKPOINT SAVED

[391/600] spambase        | linear_svm           | seed=7    | scaling_training_joint
 PASS | Macro-F1=0.916215 | BA=0.912446 | time=0.014s

[392/600] spambase        | linear_svm           | seed=7    | sampling_training_joint
 PASS | Macro-F1=0.924903 | BA=0.924494 | time=0.032s

[393/600] spambase        | logistic_regression  | seed=19   | scaling_training_joint
 PASS | Macro-F1=0.913267 | BA=0.911683 | time=0.019s

[394/600] spambase        | logistic_regression  | seed=19   | sampling_training_joint
 PASS | Macro-F1=0.917058 | BA=0.917259 | time=0.031s

[395/600] spambase        | decision_tree        | seed=19   | scaling_training_joint
 PASS | Macro-F1=0.892563 | BA=0.894658 | time=0.036s

[396/600] spambase        | decision_tree        | seed=19   | sampling_training_joint
 PASS | Macro-F1=0.883724 | BA=0.886527 | time=0.042s

[397/600] spambase        | random_forest        | seed=19   | scaling_training_joint
 PASS | Macro-F1=0.941402 | BA=0.936972 | time=0.709s

[398/600] spambase        | random_forest        | seed=19   | sampling_training_joint
 PASS | Macro-F1=0.938055 | BA=0.934284 | time=1.004s

[399/600] spambase        | linear_svm           | seed=19   | scaling_training_joint
 PASS | Macro-F1=0.912170 | BA=0.910787 | time=0.020s

[400/600] spambase        | linear_svm           | seed=19   | sampling_training_joint
 PASS | Macro-F1=0.917137 | BA=0.917740 | time=0.039s
 CHECKPOINT SAVED

[401/600] spambase        | logistic_regression  | seed=31   | scaling_training_joint
 PASS | Macro-F1=0.911624 | BA=0.907899 | time=0.048s

[402/600] spambase        | logistic_regression  | seed=31   | sampling_training_joint
 PASS | Macro-F1=0.914786 | BA=0.914986 | time=0.050s

[403/600] spambase        | decision_tree        | seed=31   | scaling_training_joint
 PASS | Macro-F1=0.868873 | BA=0.871027 | time=0.047s

[404/600] spambase        | decision_tree        | seed=31   | sampling_training_joint
 PASS | Macro-F1=0.871694 | BA=0.875707 | time=0.061s

[405/600] spambase        | random_forest        | seed=31   | scaling_training_joint
 PASS | Macro-F1=0.937201 | BA=0.935314 | time=0.635s

[406/600] spambase        | random_forest        | seed=31   | sampling_training_joint
 PASS | Macro-F1=0.931730 | BA=0.931315 | time=0.590s

[407/600] spambase        | linear_svm           | seed=31   | scaling_training_joint
 PASS | Macro-F1=0.903434 | BA=0.899219 | time=0.022s

[408/600] spambase        | linear_svm           | seed=31   | sampling_training_joint
 PASS | Macro-F1=0.912344 | BA=0.911749 | time=0.036s

[409/600] spambase        | logistic_regression  | seed=53   | scaling_training_joint
 PASS | Macro-F1=0.925047 | BA=0.925457 | time=0.031s

[410/600] spambase        | logistic_regression  | seed=53   | sampling_training_joint
 PASS | Macro-F1=0.919033 | BA=0.923931 | time=0.028s
 CHECKPOINT SAVED

[411/600] spambase        | decision_tree        | seed=53   | scaling_training_joint
 PASS | Macro-F1=0.901246 | BA=0.901826 | time=0.042s

[412/600] spambase        | decision_tree        | seed=53   | sampling_training_joint
 PASS | Macro-F1=0.888272 | BA=0.887156 | time=0.047s

[413/600] spambase        | random_forest        | seed=53   | scaling_training_joint
 PASS | Macro-F1=0.947451 | BA=0.945303 | time=0.586s

[414/600] spambase        | random_forest        | seed=53   | sampling_training_joint
 PASS | Macro-F1=0.945384 | BA=0.944955 | time=0.631s

[415/600] spambase        | linear_svm           | seed=53   | scaling_training_joint
 PASS | Macro-F1=0.924903 | BA=0.924494 | time=0.019s

[416/600] spambase        | linear_svm           | seed=53   | sampling_training_joint
 PASS | Macro-F1=0.918010 | BA=0.923517 | time=0.026s

[417/600] spambase        | logistic_regression  | seed=71   | scaling_training_joint
 PASS | Macro-F1=0.928067 | BA=0.926220 | time=0.019s

[418/600] spambase        | logistic_regression  | seed=71   | sampling_training_joint
 PASS | Macro-F1=0.929385 | BA=0.928560 | time=0.033s

[419/600] spambase        | decision_tree        | seed=71   | scaling_training_joint
 PASS | Macro-F1=0.898277 | BA=0.896183 | time=0.038s

[420/600] spambase        | decision_tree        | seed=71   | sampling_training_joint
 PASS | Macro-F1=0.889467 | BA=0.888533 | time=0.035s
 CHECKPOINT SAVED

[421/600] spambase        | random_forest        | seed=71   | scaling_training_joint
 PASS | Macro-F1=0.943938 | BA=0.941171 | time=0.571s

[422/600] spambase        | random_forest        | seed=71   | sampling_training_joint
 PASS | Macro-F1=0.949786 | BA=0.948058 | time=0.593s

[423/600] spambase        | linear_svm           | seed=71   | scaling_training_joint
 PASS | Macro-F1=0.926738 | BA=0.923880 | time=0.012s

[424/600] spambase        | linear_svm           | seed=71   | sampling_training_joint
 PASS | Macro-F1=0.930693 | BA=0.930900 | time=0.028s

[425/600] spambase        | logistic_regression  | seed=89   | scaling_training_joint
 PASS | Macro-F1=0.926584 | BA=0.922917 | time=0.019s

[426/600] spambase        | logistic_regression  | seed=89   | sampling_training_joint
 PASS | Macro-F1=0.929590 | BA=0.930004 | time=0.032s

[427/600] spambase        | decision_tree        | seed=89   | scaling_training_joint
 PASS | Macro-F1=0.899967 | BA=0.899967 | time=0.042s

[428/600] spambase        | decision_tree        | seed=89   | sampling_training_joint
 PASS | Macro-F1=0.890661 | BA=0.889911 | time=0.046s

[429/600] spambase        | random_forest        | seed=89   | scaling_training_joint
 PASS | Macro-F1=0.949579 | BA=0.946133 | time=0.576s

[430/600] spambase        | random_forest        | seed=89   | sampling_training_joint
 PASS | Macro-F1=0.948514 | BA=0.945718 | time=0.656s
 CHECKPOINT SAVED

[431/600] spambase        | linear_svm           | seed=89   | scaling_training_joint
 PASS | Macro-F1=0.925154 | BA=0.920095 | time=0.013s

[432/600] spambase        | linear_svm           | seed=89   | sampling_training_joint
 PASS | Macro-F1=0.927319 | BA=0.927730 | time=0.023s

[433/600] spambase        | logistic_regression  | seed=107  | scaling_training_joint
 PASS | Macro-F1=0.924369 | BA=0.921125 | time=0.036s

[434/600] spambase        | logistic_regression  | seed=107  | sampling_training_joint
 PASS | Macro-F1=0.912429 | BA=0.912231 | time=0.030s

[435/600] spambase        | decision_tree        | seed=107  | scaling_training_joint
 PASS | Macro-F1=0.906514 | BA=0.905344 | time=0.039s

[436/600] spambase        | decision_tree        | seed=107  | sampling_training_joint
 PASS | Macro-F1=0.895013 | BA=0.897894 | time=0.037s

[437/600] spambase        | random_forest        | seed=107  | scaling_training_joint
 PASS | Macro-F1=0.945330 | BA=0.944474 | time=0.594s

[438/600] spambase        | random_forest        | seed=107  | sampling_training_joint
 PASS | Macro-F1=0.944327 | BA=0.944541 | time=0.636s

[439/600] spambase        | linear_svm           | seed=107  | scaling_training_joint
 PASS | Macro-F1=0.909133 | BA=0.904662 | time=0.013s

[440/600] spambase        | linear_svm           | seed=107  | sampling_training_joint
 PASS | Macro-F1=0.916816 | BA=0.915815 | time=0.026s
 CHECKPOINT SAVED

[441/600] spambase        | logistic_regression  | seed=149  | scaling_training_joint
 PASS | Macro-F1=0.926148 | BA=0.926353 | time=0.016s

[442/600] spambase        | logistic_regression  | seed=149  | sampling_training_joint
 PASS | Macro-F1=0.922037 | BA=0.924694 | time=0.032s

[443/600] spambase        | decision_tree        | seed=149  | scaling_training_joint
 PASS | Macro-F1=0.884274 | BA=0.885016 | time=0.036s

[444/600] spambase        | decision_tree        | seed=149  | sampling_training_joint
 PASS | Macro-F1=0.889895 | BA=0.890459 | time=0.048s

[445/600] spambase        | random_forest        | seed=149  | scaling_training_joint
 PASS | Macro-F1=0.945110 | BA=0.942549 | time=0.581s

[446/600] spambase        | random_forest        | seed=149  | sampling_training_joint
 PASS | Macro-F1=0.948619 | BA=0.946681 | time=0.644s

[447/600] spambase        | linear_svm           | seed=149  | scaling_training_joint
 PASS | Macro-F1=0.922551 | BA=0.921739 | time=0.013s

[448/600] spambase        | linear_svm           | seed=149  | sampling_training_joint
 PASS | Macro-F1=0.924229 | BA=0.926486 | time=0.027s

[449/600] spambase        | logistic_regression  | seed=173  | scaling_training_joint
 PASS | Macro-F1=0.911529 | BA=0.907417 | time=0.017s

[450/600] spambase        | logistic_regression  | seed=173  | sampling_training_joint
 PASS | Macro-F1=0.904138 | BA=0.902589 | time=0.041s
 CHECKPOINT SAVED

[451/600] spambase        | decision_tree        | seed=173  | scaling_training_joint
 PASS | Macro-F1=0.880700 | BA=0.880884 | time=0.043s

[452/600] spambase        | decision_tree        | seed=173  | sampling_training_joint
 PASS | Macro-F1=0.878428 | BA=0.878610 | time=0.047s

[453/600] spambase        | random_forest        | seed=173  | scaling_training_joint
 PASS | Macro-F1=0.928878 | BA=0.925190 | time=0.593s

[454/600] spambase        | random_forest        | seed=173  | sampling_training_joint
 PASS | Macro-F1=0.930280 | BA=0.928012 | time=0.675s

[455/600] spambase        | linear_svm           | seed=173  | scaling_training_joint
 PASS | Macro-F1=0.906730 | BA=0.901908 | time=0.017s

[456/600] spambase        | linear_svm           | seed=173  | sampling_training_joint
 PASS | Macro-F1=0.906133 | BA=0.903418 | time=0.033s

[457/600] spambase        | logistic_regression  | seed=211  | scaling_training_joint
 PASS | Macro-F1=0.921136 | BA=0.918918 | time=0.034s

[458/600] spambase        | logistic_regression  | seed=211  | sampling_training_joint
 PASS | Macro-F1=0.926491 | BA=0.928760 | time=0.061s

[459/600] spambase        | decision_tree        | seed=211  | scaling_training_joint
 PASS | Macro-F1=0.898482 | BA=0.897146 | time=0.056s

[460/600] spambase        | decision_tree        | seed=211  | sampling_training_joint
 PASS | Macro-F1=0.903959 | BA=0.906506 | time=0.051s
 CHECKPOINT SAVED

[461/600] spambase        | random_forest        | seed=211  | scaling_training_joint
 PASS | Macro-F1=0.952943 | BA=0.948821 | time=0.928s

[462/600] spambase        | random_forest        | seed=211  | sampling_training_joint
 PASS | Macro-F1=0.954351 | BA=0.952605 | time=0.770s

[463/600] spambase        | linear_svm           | seed=211  | scaling_training_joint
 PASS | Macro-F1=0.914101 | BA=0.911135 | time=0.012s

[464/600] spambase        | linear_svm           | seed=211  | sampling_training_joint
 PASS | Macro-F1=0.917663 | BA=0.921110 | time=0.023s

[465/600] spambase        | logistic_regression  | seed=257  | scaling_training_joint
 PASS | Macro-F1=0.922776 | BA=0.923183 | time=0.016s

[466/600] spambase        | logistic_regression  | seed=257  | sampling_training_joint
 PASS | Macro-F1=0.927654 | BA=0.930137 | time=0.030s

[467/600] spambase        | decision_tree        | seed=257  | scaling_training_joint
 PASS | Macro-F1=0.895420 | BA=0.895420 | time=0.036s

[468/600] spambase        | decision_tree        | seed=257  | sampling_training_joint
 PASS | Macro-F1=0.881893 | BA=0.882261 | time=0.055s

[469/600] spambase        | random_forest        | seed=257  | scaling_training_joint
 PASS | Macro-F1=0.944052 | BA=0.942134 | time=1.144s

[470/600] spambase        | random_forest        | seed=257  | sampling_training_joint
 PASS | Macro-F1=0.946443 | BA=0.945370 | time=0.640s
 CHECKPOINT SAVED

[471/600] spambase        | linear_svm           | seed=257  | scaling_training_joint
 PASS | Macro-F1=0.908884 | BA=0.908099 | time=0.010s

[472/600] spambase        | linear_svm           | seed=257  | sampling_training_joint
 PASS | Macro-F1=0.922037 | BA=0.924694 | time=0.022s

[473/600] spambase        | logistic_regression  | seed=311  | scaling_training_joint
 PASS | Macro-F1=0.929244 | BA=0.927597 | time=0.016s

[474/600] spambase        | logistic_regression  | seed=311  | sampling_training_joint
 PASS | Macro-F1=0.924364 | BA=0.927449 | time=0.030s

[475/600] spambase        | decision_tree        | seed=311  | scaling_training_joint
 PASS | Macro-F1=0.894729 | BA=0.896450 | time=0.039s

[476/600] spambase        | decision_tree        | seed=311  | sampling_training_joint
 PASS | Macro-F1=0.886754 | BA=0.888252 | time=0.052s

[477/600] spambase        | random_forest        | seed=311  | scaling_training_joint
 PASS | Macro-F1=0.940946 | BA=0.941371 | time=0.588s

[478/600] spambase        | random_forest        | seed=311  | sampling_training_joint
 PASS | Macro-F1=0.941112 | BA=0.942815 | time=0.627s

[479/600] spambase        | linear_svm           | seed=311  | scaling_training_joint
 PASS | Macro-F1=0.919952 | BA=0.917541 | time=0.011s

[480/600] spambase        | linear_svm           | seed=311  | sampling_training_joint
 PASS | Macro-F1=0.920942 | BA=0.923798 | time=0.046s
 CHECKPOINT SAVED

[481/600] phoneme         | logistic_regression  | seed=42   | scaling_training_joint
 PASS | Macro-F1=0.660199 | BA=0.651180 | time=0.004s

[482/600] phoneme         | logistic_regression  | seed=42   | sampling_training_joint
 PASS | Macro-F1=0.695633 | BA=0.727875 | time=0.006s

[483/600] phoneme         | decision_tree        | seed=42   | scaling_training_joint
 PASS | Macro-F1=0.790416 | BA=0.783338 | time=0.015s

[484/600] phoneme         | decision_tree        | seed=42   | sampling_training_joint
 PASS | Macro-F1=0.799830 | BA=0.794379 | time=0.019s

[485/600] phoneme         | random_forest        | seed=42   | scaling_training_joint
 PASS | Macro-F1=0.851963 | BA=0.842672 | time=0.677s

[486/600] phoneme         | random_forest        | seed=42   | sampling_training_joint
 PASS | Macro-F1=0.859187 | BA=0.861515 | time=0.802s

[487/600] phoneme         | linear_svm           | seed=42   | scaling_training_joint
 PASS | Macro-F1=0.663423 | BA=0.654066 | time=0.004s

[488/600] phoneme         | linear_svm           | seed=42   | sampling_training_joint
 PASS | Macro-F1=0.695975 | BA=0.728797 | time=0.005s

[489/600] phoneme         | logistic_regression  | seed=123  | scaling_training_joint
 PASS | Macro-F1=0.690242 | BA=0.684336 | time=0.004s

[490/600] phoneme         | logistic_regression  | seed=123  | sampling_training_joint
 PASS | Macro-F1=0.701252 | BA=0.730342 | time=0.005s
 CHECKPOINT SAVED

[491/600] phoneme         | decision_tree        | seed=123  | scaling_training_joint
 PASS | Macro-F1=0.799037 | BA=0.792533 | time=0.015s

[492/600] phoneme         | decision_tree        | seed=123  | sampling_training_joint
 PASS | Macro-F1=0.799578 | BA=0.792265 | time=0.024s

[493/600] phoneme         | random_forest        | seed=123  | scaling_training_joint
 PASS | Macro-F1=0.858076 | BA=0.854786 | time=0.665s

[494/600] phoneme         | random_forest        | seed=123  | sampling_training_joint
 PASS | Macro-F1=0.854346 | BA=0.861279 | time=0.818s

[495/600] phoneme         | linear_svm           | seed=123  | scaling_training_joint
 PASS | Macro-F1=0.694435 | BA=0.689336 | time=0.004s

[496/600] phoneme         | linear_svm           | seed=123  | sampling_training_joint
 PASS | Macro-F1=0.701470 | BA=0.732456 | time=0.005s

[497/600] phoneme         | logistic_regression  | seed=2026 | scaling_training_joint
 PASS | Macro-F1=0.694330 | BA=0.681619 | time=0.004s

[498/600] phoneme         | logistic_regression  | seed=2026 | sampling_training_joint
 PASS | Macro-F1=0.732165 | BA=0.763549 | time=0.006s

[499/600] phoneme         | decision_tree        | seed=2026 | scaling_training_joint
 PASS | Macro-F1=0.828451 | BA=0.831193 | time=0.015s

[500/600] phoneme         | decision_tree        | seed=2026 | sampling_training_joint
 PASS | Macro-F1=0.795931 | BA=0.792952 | time=0.019s
 CHECKPOINT SAVED

[501/600] phoneme         | random_forest        | seed=2026 | scaling_training_joint
 PASS | Macro-F1=0.862319 | BA=0.858327 | time=0.696s

[502/600] phoneme         | random_forest        | seed=2026 | sampling_training_joint
 PASS | Macro-F1=0.865759 | BA=0.874284 | time=0.829s

[503/600] phoneme         | linear_svm           | seed=2026 | scaling_training_joint
 PASS | Macro-F1=0.696766 | BA=0.683851 | time=0.004s

[504/600] phoneme         | linear_svm           | seed=2026 | sampling_training_joint
 PASS | Macro-F1=0.728226 | BA=0.761200 | time=0.005s

[505/600] phoneme         | logistic_regression  | seed=7    | scaling_training_joint
 PASS | Macro-F1=0.659417 | BA=0.649989 | time=0.004s

[506/600] phoneme         | logistic_regression  | seed=7    | sampling_training_joint
 PASS | Macro-F1=0.711220 | BA=0.742692 | time=0.005s

[507/600] phoneme         | decision_tree        | seed=7    | scaling_training_joint
 PASS | Macro-F1=0.833302 | BA=0.828123 | time=0.015s

[508/600] phoneme         | decision_tree        | seed=7    | sampling_training_joint
 PASS | Macro-F1=0.788354 | BA=0.784411 | time=0.018s

[509/600] phoneme         | random_forest        | seed=7    | scaling_training_joint
 PASS | Macro-F1=0.882038 | BA=0.872607 | time=0.695s

[510/600] phoneme         | random_forest        | seed=7    | sampling_training_joint
 PASS | Macro-F1=0.880931 | BA=0.881986 | time=1.302s
 CHECKPOINT SAVED

[511/600] phoneme         | linear_svm           | seed=7    | scaling_training_joint
 PASS | Macro-F1=0.659417 | BA=0.649989 | time=0.006s

[512/600] phoneme         | linear_svm           | seed=7    | sampling_training_joint
 PASS | Macro-F1=0.715401 | BA=0.749269 | time=0.007s

[513/600] phoneme         | logistic_regression  | seed=19   | scaling_training_joint
 PASS | Macro-F1=0.676385 | BA=0.665611 | time=0.006s

[514/600] phoneme         | logistic_regression  | seed=19   | sampling_training_joint
 PASS | Macro-F1=0.708342 | BA=0.743111 | time=0.008s

[515/600] phoneme         | decision_tree        | seed=19   | scaling_training_joint
 PASS | Macro-F1=0.808184 | BA=0.810185 | time=0.020s

[516/600] phoneme         | decision_tree        | seed=19   | sampling_training_joint
 PASS | Macro-F1=0.819779 | BA=0.816847 | time=0.024s

[517/600] phoneme         | random_forest        | seed=19   | scaling_training_joint
 PASS | Macro-F1=0.866058 | BA=0.860022 | time=0.924s

[518/600] phoneme         | random_forest        | seed=19   | sampling_training_joint
 PASS | Macro-F1=0.877238 | BA=0.884251 | time=0.863s

[519/600] phoneme         | linear_svm           | seed=19   | scaling_training_joint
 PASS | Macro-F1=0.677216 | BA=0.666265 | time=0.004s

[520/600] phoneme         | linear_svm           | seed=19   | sampling_training_joint
 PASS | Macro-F1=0.707621 | BA=0.744570 | time=0.005s
 CHECKPOINT SAVED

[521/600] phoneme         | logistic_regression  | seed=31   | scaling_training_joint
 PASS | Macro-F1=0.664307 | BA=0.653379 | time=0.004s

[522/600] phoneme         | logistic_regression  | seed=31   | sampling_training_joint
 PASS | Macro-F1=0.713714 | BA=0.747960 | time=0.006s

[523/600] phoneme         | decision_tree        | seed=31   | scaling_training_joint
 PASS | Macro-F1=0.825871 | BA=0.821696 | time=0.015s

[524/600] phoneme         | decision_tree        | seed=31   | sampling_training_joint
 PASS | Macro-F1=0.825924 | BA=0.825924 | time=0.020s

[525/600] phoneme         | random_forest        | seed=31   | scaling_training_joint
 PASS | Macro-F1=0.863316 | BA=0.858981 | time=0.665s

[526/600] phoneme         | random_forest        | seed=31   | sampling_training_joint
 PASS | Macro-F1=0.873620 | BA=0.879519 | time=0.804s

[527/600] phoneme         | linear_svm           | seed=31   | scaling_training_joint
 PASS | Macro-F1=0.673314 | BA=0.661651 | time=0.004s

[528/600] phoneme         | linear_svm           | seed=31   | sampling_training_joint
 PASS | Macro-F1=0.715830 | BA=0.752306 | time=0.005s

[529/600] phoneme         | logistic_regression  | seed=53   | scaling_training_joint
 PASS | Macro-F1=0.692242 | BA=0.681115 | time=0.004s

[530/600] phoneme         | logistic_regression  | seed=53   | sampling_training_joint
 PASS | Macro-F1=0.739536 | BA=0.776050 | time=0.006s
 CHECKPOINT SAVED

[531/600] phoneme         | decision_tree        | seed=53   | scaling_training_joint
 PASS | Macro-F1=0.810406 | BA=0.806997 | time=0.015s

[532/600] phoneme         | decision_tree        | seed=53   | sampling_training_joint
 PASS | Macro-F1=0.781202 | BA=0.775602 | time=0.018s

[533/600] phoneme         | random_forest        | seed=53   | scaling_training_joint
 PASS | Macro-F1=0.864723 | BA=0.855408 | time=0.688s

[534/600] phoneme         | random_forest        | seed=53   | sampling_training_joint
 PASS | Macro-F1=0.869048 | BA=0.871096 | time=0.823s

[535/600] phoneme         | linear_svm           | seed=53   | scaling_training_joint
 PASS | Macro-F1=0.699976 | BA=0.689002 | time=0.004s

[536/600] phoneme         | linear_svm           | seed=53   | sampling_training_joint
 PASS | Macro-F1=0.735842 | BA=0.774623 | time=0.005s

[537/600] phoneme         | logistic_regression  | seed=71   | scaling_training_joint
 PASS | Macro-F1=0.650293 | BA=0.640374 | time=0.004s

[538/600] phoneme         | logistic_regression  | seed=71   | sampling_training_joint
 PASS | Macro-F1=0.710004 | BA=0.749839 | time=0.006s

[539/600] phoneme         | decision_tree        | seed=71   | scaling_training_joint
 PASS | Macro-F1=0.822704 | BA=0.820924 | time=0.016s

[540/600] phoneme         | decision_tree        | seed=71   | sampling_training_joint
 PASS | Macro-F1=0.815240 | BA=0.806964 | time=0.020s
 CHECKPOINT SAVED

[541/600] phoneme         | random_forest        | seed=71   | scaling_training_joint
 PASS | Macro-F1=0.883079 | BA=0.880258 | time=0.732s

[542/600] phoneme         | random_forest        | seed=71   | sampling_training_joint
 PASS | Macro-F1=0.881826 | BA=0.889637 | time=0.835s

[543/600] phoneme         | linear_svm           | seed=71   | scaling_training_joint
 PASS | Macro-F1=0.638859 | BA=0.629183 | time=0.004s

[544/600] phoneme         | linear_svm           | seed=71   | sampling_training_joint
 PASS | Macro-F1=0.708075 | BA=0.749721 | time=0.005s

[545/600] phoneme         | logistic_regression  | seed=89   | scaling_training_joint
 PASS | Macro-F1=0.674722 | BA=0.667372 | time=0.004s

[546/600] phoneme         | logistic_regression  | seed=89   | sampling_training_joint
 PASS | Macro-F1=0.683767 | BA=0.730742 | time=0.005s

[547/600] phoneme         | decision_tree        | seed=89   | scaling_training_joint
 PASS | Macro-F1=0.804179 | BA=0.803339 | time=0.015s

[548/600] phoneme         | decision_tree        | seed=89   | sampling_training_joint
 PASS | Macro-F1=0.775503 | BA=0.775249 | time=0.019s

[549/600] phoneme         | random_forest        | seed=89   | scaling_training_joint
 PASS | Macro-F1=0.841395 | BA=0.836396 | time=0.720s

[550/600] phoneme         | random_forest        | seed=89   | sampling_training_joint
 PASS | Macro-F1=0.865759 | BA=0.874284 | time=0.836s
 CHECKPOINT SAVED

[551/600] phoneme         | linear_svm           | seed=89   | scaling_training_joint
 PASS | Macro-F1=0.684116 | BA=0.676299 | time=0.004s

[552/600] phoneme         | linear_svm           | seed=89   | sampling_training_joint
 PASS | Macro-F1=0.683765 | BA=0.732856 | time=0.006s

[553/600] phoneme         | logistic_regression  | seed=107  | scaling_training_joint
 PASS | Macro-F1=0.654555 | BA=0.645257 | time=0.005s

[554/600] phoneme         | logistic_regression  | seed=107  | sampling_training_joint
 PASS | Macro-F1=0.710715 | BA=0.742960 | time=0.006s

[555/600] phoneme         | decision_tree        | seed=107  | scaling_training_joint
 PASS | Macro-F1=0.815972 | BA=0.814229 | time=0.015s

[556/600] phoneme         | decision_tree        | seed=107  | sampling_training_joint
 PASS | Macro-F1=0.806055 | BA=0.804648 | time=0.020s

[557/600] phoneme         | random_forest        | seed=107  | scaling_training_joint
 PASS | Macro-F1=0.865316 | BA=0.860290 | time=0.715s

[558/600] phoneme         | random_forest        | seed=107  | sampling_training_joint
 PASS | Macro-F1=0.865351 | BA=0.866364 | time=0.787s

[559/600] phoneme         | linear_svm           | seed=107  | scaling_training_joint
 PASS | Macro-F1=0.658606 | BA=0.648798 | time=0.004s

[560/600] phoneme         | linear_svm           | seed=107  | sampling_training_joint
 PASS | Macro-F1=0.706517 | BA=0.739688 | time=0.005s
 CHECKPOINT SAVED

[561/600] phoneme         | logistic_regression  | seed=149  | scaling_training_joint
 PASS | Macro-F1=0.673366 | BA=0.661115 | time=0.004s

[562/600] phoneme         | logistic_regression  | seed=149  | sampling_training_joint
 PASS | Macro-F1=0.724919 | BA=0.760696 | time=0.006s

[563/600] phoneme         | decision_tree        | seed=149  | scaling_training_joint
 PASS | Macro-F1=0.799879 | BA=0.800989 | time=0.015s

[564/600] phoneme         | decision_tree        | seed=149  | sampling_training_joint
 PASS | Macro-F1=0.796912 | BA=0.796912 | time=0.019s

[565/600] phoneme         | random_forest        | seed=149  | scaling_training_joint
 PASS | Macro-F1=0.876098 | BA=0.872640 | time=1.058s

[566/600] phoneme         | random_forest        | seed=149  | sampling_training_joint
 PASS | Macro-F1=0.870661 | BA=0.877556 | time=1.264s

[567/600] phoneme         | linear_svm           | seed=149  | scaling_training_joint
 PASS | Macro-F1=0.688900 | BA=0.675696 | time=0.004s

[568/600] phoneme         | linear_svm           | seed=149  | sampling_training_joint
 PASS | Macro-F1=0.727777 | BA=0.763582 | time=0.005s

[569/600] phoneme         | logistic_regression  | seed=173  | scaling_training_joint
 PASS | Macro-F1=0.660013 | BA=0.652522 | time=0.005s

[570/600] phoneme         | logistic_regression  | seed=173  | sampling_training_joint
 PASS | Macro-F1=0.686304 | BA=0.725172 | time=0.006s
 CHECKPOINT SAVED

[571/600] phoneme         | decision_tree        | seed=173  | scaling_training_joint
 PASS | Macro-F1=0.800757 | BA=0.812482 | time=0.016s

[572/600] phoneme         | decision_tree        | seed=173  | sampling_training_joint
 PASS | Macro-F1=0.786147 | BA=0.786676 | time=0.018s

[573/600] phoneme         | random_forest        | seed=173  | scaling_training_joint
 PASS | Macro-F1=0.859691 | BA=0.863360 | time=0.706s

[574/600] phoneme         | random_forest        | seed=173  | sampling_training_joint
 PASS | Macro-F1=0.854855 | BA=0.866162 | time=0.876s

[575/600] phoneme         | linear_svm           | seed=173  | scaling_training_joint
 PASS | Macro-F1=0.656916 | BA=0.649099 | time=0.005s

[576/600] phoneme         | linear_svm           | seed=173  | sampling_training_joint
 PASS | Macro-F1=0.689115 | BA=0.728058 | time=0.006s

[577/600] phoneme         | logistic_regression  | seed=211  | scaling_training_joint
 PASS | Macro-F1=0.679141 | BA=0.672372 | time=0.005s

[578/600] phoneme         | logistic_regression  | seed=211  | sampling_training_joint
 PASS | Macro-F1=0.699408 | BA=0.746482 | time=0.006s

[579/600] phoneme         | decision_tree        | seed=211  | scaling_training_joint
 PASS | Macro-F1=0.813290 | BA=0.816494 | time=0.015s

[580/600] phoneme         | decision_tree        | seed=211  | sampling_training_joint
 PASS | Macro-F1=0.799815 | BA=0.804295 | time=0.020s
 CHECKPOINT SAVED

[581/600] phoneme         | random_forest        | seed=211  | scaling_training_joint
 PASS | Macro-F1=0.865840 | BA=0.868210 | time=0.708s

[582/600] phoneme         | random_forest        | seed=211  | sampling_training_joint
 PASS | Macro-F1=0.870780 | BA=0.881516 | time=0.829s

[583/600] phoneme         | linear_svm           | seed=211  | scaling_training_joint
 PASS | Macro-F1=0.683284 | BA=0.675645 | time=0.004s

[584/600] phoneme         | linear_svm           | seed=211  | sampling_training_joint
 PASS | Macro-F1=0.699956 | BA=0.748328 | time=0.006s

[585/600] phoneme         | logistic_regression  | seed=257  | scaling_training_joint
 PASS | Macro-F1=0.685630 | BA=0.677876 | time=0.005s

[586/600] phoneme         | logistic_regression  | seed=257  | sampling_training_joint
 PASS | Macro-F1=0.712220 | BA=0.742155 | time=0.006s

[587/600] phoneme         | decision_tree        | seed=257  | scaling_training_joint
 PASS | Macro-F1=0.815624 | BA=0.813306 | time=0.019s

[588/600] phoneme         | decision_tree        | seed=257  | sampling_training_joint
 PASS | Macro-F1=0.810302 | BA=0.810302 | time=0.018s

[589/600] phoneme         | random_forest        | seed=257  | scaling_training_joint
 PASS | Macro-F1=0.855784 | BA=0.860742 | time=0.697s

[590/600] phoneme         | random_forest        | seed=257  | sampling_training_joint
 PASS | Macro-F1=0.859142 | BA=0.873662 | time=0.831s
 CHECKPOINT SAVED

[591/600] phoneme         | linear_svm           | seed=257  | scaling_training_joint
 PASS | Macro-F1=0.687140 | BA=0.679454 | time=0.004s

[592/600] phoneme         | linear_svm           | seed=257  | sampling_training_joint
 PASS | Macro-F1=0.710193 | BA=0.739923 | time=0.005s

[593/600] phoneme         | logistic_regression  | seed=311  | scaling_training_joint
 PASS | Macro-F1=0.661833 | BA=0.650879 | time=0.004s

[594/600] phoneme         | logistic_regression  | seed=311  | sampling_training_joint
 PASS | Macro-F1=0.699453 | BA=0.730224 | time=0.005s

[595/600] phoneme         | decision_tree        | seed=311  | scaling_training_joint
 PASS | Macro-F1=0.800771 | BA=0.795033 | time=0.014s

[596/600] phoneme         | decision_tree        | seed=311  | sampling_training_joint
 PASS | Macro-F1=0.784330 | BA=0.779679 | time=0.018s

[597/600] phoneme         | random_forest        | seed=311  | scaling_training_joint
 PASS | Macro-F1=0.871556 | BA=0.871214 | time=0.743s

[598/600] phoneme         | random_forest        | seed=311  | sampling_training_joint
 PASS | Macro-F1=0.872307 | BA=0.880979 | time=0.832s

[599/600] phoneme         | linear_svm           | seed=311  | scaling_training_joint
 PASS | Macro-F1=0.657666 | BA=0.646801 | time=0.004s

[600/600] phoneme         | linear_svm           | seed=311  | sampling_training_joint
 PASS | Macro-F1=0.703487 | BA=0.734688 | time=0.006s
 CHECKPOINT SAVED

==============================================================================
BATCH 3A NEW JOINT-RUN VALIDATION
==============================================================================
{
  "G1_expected_600_rows": true,
  "G2_unique_run_keys": true,
  "G3_all_runs_pass": true,
  "G4_metric_integrity": true,
  "G5_configuration_accounting": {
    "sampling_training_joint": 300,
    "scaling_training_joint": 300
  },
  "G6_model_accounting": {
    "decision_tree": 150,
    "linear_svm": 150,
    "logistic_regression": 150,
    "random_forest": 150
  },
  "G7_dataset_accounting": {
    "bank_marketing": 120,
    "breast_cancer": 120,
    "phoneme": 120,
    "pima_diabetes": 120,
    "spambase": 120
  },
  "G8_seed_accounting": {
    "7": 40,
    "19": 40,
    "31": 40,
    "42": 40,
    "53": 40,
    "71": 40,
    "89": 40,
    "107": 40,
    "123": 40,
    "149": 40,
    "173": 40,
    "211": 40,
    "257": 40,
    "311": 40,
    "2026": 40
  },
  "G9_exact_split_reuse": true,
  "G10_manifest_key_alignment": true,
  "pass_runs": 600,
  "fail_runs": 0,
  "production_authorized": true,
  "created_at": "2026-09-27T08:34:17.003404"
}

==============================================================================
BATCH 3A INTERACTION ASSEMBLY
==============================================================================
Base/single rows reused : 1500
New joint rows          : 600
Combined rows           : 2100
Interaction rows        : 600
Maximum arithmetic err  : 0.0

Interaction summary:
                     count      mean       std       min       max
interaction                                                       
sampling_x_training    300  0.001583  0.022943 -0.083326  0.112155
scaling_x_training     300 -0.000717  0.013853 -0.101023  0.055254

BATCH 3A: COMPLETE AND VALIDATED.
New results : /content/CTD_BATCH3A_SCALING_SAMPLING_TRAINING_PRODUCTION/CTD_BATCH3A_NEW_JOINT_RESULTS.csv
Interactions: /content/CTD_BATCH3A_SCALING_SAMPLING_TRAINING_PRODUCTION/CTD_BATCH3A_INTERACTION_VALUES.csv
Final report: /content/CTD_BATCH3A_SCALING_SAMPLING_TRAINING_PRODUCTION/CTD_BATCH3A_FINAL_VALIDATION_REPORT.json

==============================================================================
BATCH 3A v4 EXECUTION RETURNED
==============================================================================

```