# CTD IJDSA Major Revision — Batch 0

Locked design:
- 6 datasets
- 4 models
- 15 seeds
- 4 single-decision interventions
- 4 principal interaction pairs
- 9 configurations per dataset/model/seed
- 3,240 expected production fits

Principal interaction pairs:
1. Scaling × Training Fraction
2. Sampling × Training Fraction
3. Sampling × Representation
4. Representation × Training Fraction

The original No Scaling × PCA95 interaction is auxiliary only.

Candidate practical thresholds: [0.005, 0.01, 0.02]
Primary planning threshold: 0.01

Run the engine validation before any real experiment:
python CTD_IJDSA_REVISION_BATCH0_ENGINE_VALIDATION_v1_0.py
