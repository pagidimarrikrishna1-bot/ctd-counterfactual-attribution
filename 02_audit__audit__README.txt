CTD IJDSA — Figures 2–5 package
================================

This package contains a Colab-ready script that:

1. Reads the audited Batch 3A interaction and joint-result ledgers.
2. Reads the five authoritative base/single-condition ledgers.
3. Independently recomputes the two Batch 3A interaction families.
4. Recomputes the 900 pairwise model comparisons.
5. Generates new Figures 2–5 at 600 dpi.
6. Inserts Figures 2–5 into the current revised manuscript after the relevant Results subsections.
7. Saves a new manuscript copy and a figure-generation audit report.

Figures:
- Figure 2: Distribution of Scaling × Training and Sampling × Training interactions.
- Figure 3: Dataset-level heterogeneity.
- Figure 4: Pairwise model-order reversal rates.
- Figure 5: Practical-effect filtered reversals at 0.005, 0.010, 0.020.

Important:
- Figure 1 is retained.
- Obsolete empirical Figures 2–5 from the old manuscript are not reused.
- No ML model is retrained.
- No scientific result is hard-coded into the plots except validation checks against already audited counts.
- Run the script in the same Colab environment containing the Batch 3A artifacts and current manuscript.

The generated manuscript is intended for internal visual review. Springer Nature's general submission guidance says figures should be uploaded as separate files; therefore, keep the PNG files and upload them separately in the revision system according to the journal's current submission interface.
