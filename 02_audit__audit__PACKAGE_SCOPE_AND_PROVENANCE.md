# CTD IJDSA Reproducibility Package — Scope and Provenance

This package accompanies the revised manuscript **Counterfactual Attribution of Machine-Learning Performance to Experimental Decisions** and its point-by-point response to the IJDSA major-revision comments.

## Evidence represented

The revised manuscript reports an audited evidence base of:

- five production datasets: Breast Cancer Wisconsin, Bank Marketing, Pima Indians Diabetes, Spambase, and Phoneme (OpenML 1489);
- four classical classifiers: Logistic Regression, Decision Tree, Random Forest, and Linear SVM;
- fifteen prespecified seeds;
- 2,100 combined evidence observations;
- 600 newly generated joint-intervention observations;
- 600 interaction estimates;
- 900 pairwise model comparisons.

The completed production interaction families are Scaling × Training Fraction and Sampling × Training Fraction. Covertype is retained only as a resource-validation case and is not part of the five-dataset production interaction results.

## Package contents

- `scripts/` — experiment validation/production and figure-generation scripts used in the revision workflow.
- `manifests/` — frozen experiment manifest material.
- `figures/` — revised empirical Figures 2–5.
- `audit/` — provenance, production/figure execution logs, and figure cleanup audit material.
- `docs/` — the revised manuscript and point-by-point response supplied for the revision, plus audited PDF reference copies.
- `requirements.txt` — core Python environment versions recorded for the production/validation workflow.

## Data

Raw benchmark datasets are not redistributed in this repository. The experiment scripts identify the required public datasets and their acquisition identifiers. Users should obtain the datasets from their original public sources/OpenML and follow the preprocessing and split specifications in the scripts and manuscript.

## Reproduction scope

The package is intended to document and support reproduction of the revised experimental protocol. Some production runs are computationally substantial. The repository therefore preserves the exact scripts, manifests, provenance records, and figure-generation procedure rather than redistributing large raw datasets or private execution environments.

## Interpretation boundary

The manuscript explicitly treats CTD as a complementary decision-level attribution and consequence-analysis framework. CDE and interaction quantities are controlled differences between experimental configurations under the specified protocol; they are not unrestricted real-world causal effects. Ranking-reversal frequencies are benchmark-specific and should not be interpreted as population probabilities for arbitrary machine-learning studies.

## Associated manuscript

The public repository created for this revision is:

https://github.com/pagidimarrikrishna1-bot/ctd-counterfactual-attribution
