
# CTD IJDSA — FIGURES 2–5 GENERATION + MANUSCRIPT INSERTION
# ==========================================================
# Purpose:
#   1) Read the audited Batch 3A evidence already produced in Colab.
#   2) Independently recompute the two interaction families.
#   3) Generate new Figures 2–5 at 600 dpi.
#   4) Insert Figures 2–5 into the current revised manuscript.
#   5) Preserve Figure 1; do NOT reuse obsolete empirical Figures 2–5.
#
# IMPORTANT:
#   Run this only after the Batch 3A artifacts exist at the paths below.
#   No ML models are retrained by this script.
#   No scientific results are invented or hard-coded into the figures.

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Inches, Pt
from docx.oxml import OxmlElement
from docx.text.paragraph import Paragraph
from docx.table import Table

ROOT = Path("/content/CTD_BATCH3A_SCALING_SAMPLING_TRAINING_PRODUCTION")
MANUSCRIPT = Path("/content/CTD_IJDSA_REVISED_MANUSCRIPT_FINAL_ASSEMBLY_v1.docx")
OUT = ROOT / "CTD_IJDSA_FIGURES_2_TO_5"
OUT.mkdir(parents=True, exist_ok=True)

INTERACTIONS = ROOT / "CTD_BATCH3A_INTERACTION_VALUES.csv"
JOINT = ROOT / "CTD_BATCH3A_NEW_JOINT_RESULTS.csv"

BASE_LEDGER_PATHS = {
    "breast_cancer": Path("/content/CTD_BATCH1_LEDGER/CTD_BATCH1A_RESULTS_LEDGER.csv"),
    "bank_marketing": Path("/content/CTD_BATCH1B_R1_PRODUCTION/CTD_BATCH1B_R1_RESULTS_LEDGER.csv"),
    "pima_diabetes": Path("/content/CTD_BATCH2A_PIMA_PRODUCTION/CTD_BATCH2A_PIMA_RESULTS_LEDGER.csv"),
    "spambase": Path("/content/CTD_BATCH2B_SPAMBASE_PRODUCTION/CTD_BATCH2B_SPAMBASE_RESULTS_LEDGER.csv"),
    "phoneme": Path("/content/CTD_BATCH2C_PHONEME_PRODUCTION/CTD_BATCH2C_PHONEME_RESULTS_LEDGER.csv"),
}

DATASET_LABELS = {
    "bank_marketing": "Bank Marketing",
    "breast_cancer": "Breast Cancer",
    "phoneme": "Phoneme",
    "pima_diabetes": "Pima Diabetes",
    "spambase": "Spambase",
}

def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

# -------------------- Evidence loading --------------------
require_file(MANUSCRIPT)
require_file(INTERACTIONS)
require_file(JOINT)
for p in BASE_LEDGER_PATHS.values():
    require_file(p)

inter = pd.read_csv(INTERACTIONS)
joint = pd.read_csv(JOINT)

required_i = {"dataset", "model", "seed", "interaction", "interaction_value"}
missing_i = required_i - set(inter.columns)
if missing_i:
    raise RuntimeError(f"Interaction ledger missing columns: {sorted(missing_i)}")

if len(inter) != 600:
    raise RuntimeError(f"Expected 600 interaction rows; found {len(inter)}")

expected_interaction_counts = {
    "scaling_x_training": 300,
    "sampling_x_training": 300,
}
if inter["interaction"].value_counts().to_dict() != expected_interaction_counts:
    raise RuntimeError(
        "Unexpected interaction-family accounting: "
        + str(inter["interaction"].value_counts().to_dict())
    )

if not np.isfinite(inter["interaction_value"]).all():
    raise RuntimeError("Non-finite interaction values found.")

base_parts = []
for ds, path in BASE_LEDGER_PATHS.items():
    z = pd.read_csv(path)
    z = z[
        (z["status"].astype(str) == "PASS")
        & (z["dataset"].astype(str) == ds)
    ].copy()
    z = z[
        z["configuration"].isin(
            [
                "baseline",
                "scaling_cf",
                "sampling_cf",
                "representation_cf",
                "training_fraction_cf",
            ]
        )
    ].copy()

    if len(z) != 300:
        raise RuntimeError(f"{ds}: expected 300 base rows; found {len(z)}")

    base_parts.append(z)

base = pd.concat(base_parts, ignore_index=True)

joint = joint[joint["status"].astype(str) == "PASS"].copy()
if len(joint) != 600:
    raise RuntimeError(f"Expected 600 PASS joint rows; found {len(joint)}")

combined = pd.concat([base, joint], ignore_index=True)

if len(combined) != 2100:
    raise RuntimeError(f"Expected 2100 combined rows; found {len(combined)}")

if combined.duplicated(
    ["dataset", "model", "seed", "configuration"]
).any():
    raise RuntimeError("Duplicate experiment identities found.")

# -------------------- Independent interaction recomputation --------------------
wide = (
    combined.pivot_table(
        index=["dataset", "model", "seed"],
        columns="configuration",
        values="macro_f1",
        aggfunc="first",
    )
    .reset_index()
)

required_configs = [
    "baseline",
    "scaling_cf",
    "sampling_cf",
    "training_fraction_cf",
    "scaling_training_joint",
    "sampling_training_joint",
]
for cfg in required_configs:
    if cfg not in wide.columns:
        raise RuntimeError(f"Missing configuration in assembled evidence: {cfg}")

wide["scaling_x_training_calc"] = (
    wide["scaling_training_joint"]
    - wide["scaling_cf"]
    - wide["training_fraction_cf"]
    + wide["baseline"]
)

wide["sampling_x_training_calc"] = (
    wide["sampling_training_joint"]
    - wide["sampling_cf"]
    - wide["training_fraction_cf"]
    + wide["baseline"]
)

check = inter.merge(
    wide[
        [
            "dataset",
            "model",
            "seed",
            "scaling_x_training_calc",
            "sampling_x_training_calc",
        ]
    ],
    on=["dataset", "model", "seed"],
    how="left",
)

calculated = np.where(
    check["interaction"].eq("sampling_x_training"),
    check["sampling_x_training_calc"],
    check["scaling_x_training_calc"],
)

max_interaction_error = float(
    np.max(np.abs(check["interaction_value"].to_numpy() - calculated))
)

if max_interaction_error > 1e-12:
    raise RuntimeError(
        f"Independent interaction recomputation failed: "
        f"max error={max_interaction_error}"
    )

# -------------------- Ranking evidence --------------------
rank = (
    combined.pivot_table(
        index=["dataset", "model", "seed"],
        columns="configuration",
        values="macro_f1",
        aggfunc="first",
    )
    .reset_index()
)

pair_records = []

for (dataset, seed), group in rank.groupby(["dataset", "seed"]):
    if len(group) != 4:
        raise RuntimeError(
            f"{dataset}, seed {seed}: expected four models; found {len(group)}"
        )

    models = group["model"].tolist()

    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            model_a = models[i]
            model_b = models[j]

            row_a = group.loc[group["model"] == model_a].iloc[0]
            row_b = group.loc[group["model"] == model_b].iloc[0]

            baseline_gap = float(row_a["baseline"] - row_b["baseline"])

            for condition, joint_cfg in [
                ("scaling_x_training", "scaling_training_joint"),
                ("sampling_x_training", "sampling_training_joint"),
            ]:
                counterfactual_gap = float(
                    row_a[joint_cfg] - row_b[joint_cfg]
                )

                reversal = (baseline_gap * counterfactual_gap) < 0

                pair_records.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "model_a": model_a,
                        "model_b": model_b,
                        "condition": condition,
                        "baseline_gap": baseline_gap,
                        "counterfactual_gap": counterfactual_gap,
                        "reversal": bool(reversal),
                        "minimum_gap": min(
                            abs(baseline_gap),
                            abs(counterfactual_gap),
                        ),
                    }
                )

pairs = pd.DataFrame(pair_records)

if len(pairs) != 900:
    raise RuntimeError(f"Expected 900 pairwise comparisons; found {len(pairs)}")

expected_raw_reversals = {
    "scaling_x_training": 73,
    "sampling_x_training": 110,
}

for condition, expected in expected_raw_reversals.items():
    observed = int(
        pairs.loc[pairs["condition"] == condition, "reversal"].sum()
    )
    if observed != expected:
        raise RuntimeError(
            f"{condition}: expected {expected} raw reversals; found {observed}"
        )

# -------------------- Figure formatting --------------------
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 9

# Figure 2 — interaction distributions
groups = [
    inter.loc[
        inter["interaction"] == "scaling_x_training",
        "interaction_value",
    ].to_numpy(),
    inter.loc[
        inter["interaction"] == "sampling_x_training",
        "interaction_value",
    ].to_numpy(),
]

fig, ax = plt.subplots(figsize=(7.2, 4.3))

ax.violinplot(
    groups,
    positions=[1, 2],
    showmeans=True,
    showmedians=True,
    showextrema=True,
)

for i, values in enumerate(groups, start=1):
    rng = np.random.default_rng(100 + i)
    jitter = rng.uniform(-0.055, 0.055, size=len(values))
    ax.scatter(
        i + jitter,
        values,
        s=7,
        alpha=0.25,
    )

ax.axhline(0, linewidth=1, linestyle="--")
ax.set_xticks(
    [1, 2],
    ["Scaling × Training", "Sampling × Training"],
)
ax.set_ylabel("Interaction effect (Macro-F1)")
ax.set_title("Distribution of joint decision interactions")
ax.grid(axis="y", alpha=0.2)

fig.tight_layout()
fig2 = OUT / "Figure_2_Interaction_Distributions.png"
fig.savefig(fig2, dpi=600, bbox_inches="tight")
plt.close(fig)

# Figure 3 — dataset heterogeneity
dataset_order = [
    "bank_marketing",
    "breast_cancer",
    "phoneme",
    "pima_diabetes",
    "spambase",
]

dataset_stats = (
    inter.groupby(["interaction", "dataset"])["interaction_value"]
    .agg(mean="mean", std="std")
    .reset_index()
)

fig, ax = plt.subplots(figsize=(7.2, 4.5))

x = np.arange(len(dataset_order))
width = 0.36

for j, condition in enumerate(
    ["scaling_x_training", "sampling_x_training"]
):
    q = (
        dataset_stats[
            dataset_stats["interaction"] == condition
        ]
        .set_index("dataset")
        .reindex(dataset_order)
    )

    label = (
        "Scaling × Training"
        if condition == "scaling_x_training"
        else "Sampling × Training"
    )

    ax.errorbar(
        x + (j - 0.5) * width,
        q["mean"],
        yerr=q["std"],
        fmt="o",
        capsize=3,
        linewidth=1.2,
        label=label,
    )

ax.axhline(0, linewidth=1, linestyle="--")
ax.set_xticks(
    x,
    [DATASET_LABELS[d] for d in dataset_order],
    rotation=20,
    ha="right",
)
ax.set_ylabel("Mean interaction ± SD (Macro-F1)")
ax.set_title("Dataset-level heterogeneity of interaction effects")
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.2)

fig.tight_layout()
fig3 = OUT / "Figure_3_Dataset_Heterogeneity.png"
fig.savefig(fig3, dpi=600, bbox_inches="tight")
plt.close(fig)

# Figure 4 — pairwise model-order reversal rates
rates = (
    pairs.groupby("condition")["reversal"]
    .mean()
    .mul(100)
    .reindex(["scaling_x_training", "sampling_x_training"])
)

fig, ax = plt.subplots(figsize=(7.0, 4.1))

bars = ax.bar([0, 1], rates.values)

ax.set_xticks(
    [0, 1],
    ["Scaling × Training", "Sampling × Training"],
)
ax.set_ylabel("Pairwise reversal rate (%)")
ax.set_title("Model-order reversals under joint interventions")
ax.set_ylim(0, max(rates.values) * 1.25)

for bar, value in zip(bars, rates.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{value:.2f}%",
        ha="center",
    )

ax.grid(axis="y", alpha=0.2)

fig.tight_layout()
fig4 = OUT / "Figure_4_Model_Ranking_Reversals.png"
fig.savefig(fig4, dpi=600, bbox_inches="tight")
plt.close(fig)

# Figure 5 — practical-effect reversal counts
thresholds = [0.005, 0.010, 0.020]

threshold_values = {}
for threshold in thresholds:
    threshold_values[threshold] = [
        int(
            (
                (pairs["condition"] == condition)
                & pairs["reversal"]
                & (pairs["minimum_gap"] >= threshold)
            ).sum()
        )
        for condition in [
            "scaling_x_training",
            "sampling_x_training",
        ]
    ]

fig, ax = plt.subplots(figsize=(7.2, 4.2))

x = np.arange(len(thresholds))
width = 0.34

scaling_values = [
    threshold_values[t][0] for t in thresholds
]
sampling_values = [
    threshold_values[t][1] for t in thresholds
]

bars1 = ax.bar(
    x - width / 2,
    scaling_values,
    width,
    label="Scaling × Training",
)

bars2 = ax.bar(
    x + width / 2,
    sampling_values,
    width,
    label="Sampling × Training",
)

ax.set_xticks(
    x,
    ["0.005", "0.010", "0.020"],
)
ax.set_xlabel("Minimum absolute model-performance gap")
ax.set_ylabel("Pairwise reversals")
ax.set_title("Practical-effect filtering of model-order reversals")
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.2)

for bars in [bars1, bars2]:
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{int(bar.get_height())}",
            ha="center",
        )

fig.tight_layout()
fig5 = OUT / "Figure_5_Practical_Effect_Reversals.png"
fig.savefig(fig5, dpi=600, bbox_inches="tight")
plt.close(fig)

# -------------------- Manuscript insertion --------------------
# The current v1 manuscript contains Figure 1 but no obsolete empirical
# Figures 2–5. We insert the new figures after the first substantive
# paragraph of each corresponding Results subsection.

doc = Document(str(MANUSCRIPT))

figure_specs = [
    (
        "5.2 Interaction Effects",
        fig2,
        "Figure 2. Distribution of the two completed joint-intervention interaction effects across the 300 dataset–model–seed cells for each interaction family.",
    ),
    (
        "5.3 Cross-Dataset and Model Heterogeneity",
        fig3,
        "Figure 3. Dataset-level heterogeneity of the two completed joint-intervention effects. Points show dataset-level means and error bars show one standard deviation across dataset–model–seed observations.",
    ),
    (
        "5.4 Model-Ranking Consequences",
        fig4,
        "Figure 4. Pairwise model-order reversal rates under the two completed joint interventions. Reversals are defined by a sign change in the pairwise model-performance gap between baseline and the corresponding joint condition.",
    ),
    (
        "5.5 Practical-Effect Ranking Reversals",
        fig5,
        "Figure 5. Pairwise model-order reversals remaining after practical-effect filtering at minimum absolute Macro-F1 gaps of 0.005, 0.010, and 0.020. These thresholds are interpretive filters rather than statistical significance thresholds.",
    ),
]

def all_blocks(document):
    blocks = []
    for child in document.element.body.iterchildren():
        if child.tag.endswith("}p"):
            blocks.append(Paragraph(child, document))
        elif child.tag.endswith("}tbl"):
            blocks.append(Table(child, document))
    return blocks

def insert_figure_after(paragraph, image_path, caption_text):
    p = paragraph._p

    image_p = OxmlElement("w:p")
    p.addnext(image_p)
    image_paragraph = Paragraph(image_p, paragraph._parent)

    run = image_paragraph.add_run()
    run.add_picture(str(image_path), width=Inches(6.55))

    caption_p = OxmlElement("w:p")
    image_p.addnext(caption_p)
    caption_paragraph = Paragraph(
        caption_p,
        paragraph._parent,
    )
    caption_paragraph.alignment = 1

    caption_run = caption_paragraph.add_run(caption_text)
    caption_run.font.name = "Times New Roman"
    caption_run.font.size = Pt(10)

for heading, image_path, caption in figure_specs:
    blocks = all_blocks(doc)

    heading_index = None
    for i, block in enumerate(blocks):
        if isinstance(block, Paragraph) and block.text.strip() == heading:
            heading_index = i
            break

    if heading_index is None:
        raise RuntimeError(
            f"Could not locate manuscript subsection heading: {heading}"
        )

    inserted = False

    for block in blocks[heading_index + 1 :]:
        if isinstance(block, Paragraph):
            text = block.text.strip()

            if not text:
                continue

            # Insert after the first substantive paragraph in the subsection.
            # Tables are skipped automatically.
            if not text.startswith("Table "):
                insert_figure_after(
                    block,
                    image_path,
                    caption,
                )
                inserted = True
                break

    if not inserted:
        raise RuntimeError(
            f"Could not find insertion point for {heading}"
        )

# Save a separate review copy with figures embedded.
embedded_manuscript = (
    ROOT / "CTD_IJDSA_REVISED_MANUSCRIPT_WITH_NEW_FIGURES_2_TO_5.docx"
)
doc.save(embedded_manuscript)

# -------------------- Audit report --------------------
audit = pd.DataFrame(
    [
        [
            "Figure 2",
            fig2.name,
            600,
            "600 interaction observations; independent arithmetic check",
        ],
        [
            "Figure 3",
            fig3.name,
            600,
            "600 interaction observations grouped by dataset",
        ],
        [
            "Figure 4",
            fig4.name,
            600,
            "900 pairwise comparisons; 73 and 110 raw reversals",
        ],
        [
            "Figure 5",
            fig5.name,
            600,
            "900 pairwise comparisons; thresholds 0.005, 0.010, 0.020",
        ],
    ],
    columns=[
        "Figure",
        "File",
        "DPI",
        "Evidence basis",
    ],
)

audit.to_csv(
    OUT / "FIGURE_GENERATION_REPORT.csv",
    index=False,
)

print("=" * 78)
print("CTD IJDSA — FIGURES 2–5 GENERATION + INSERTION")
print("=" * 78)
print("Independent interaction arithmetic max error:", max_interaction_error)
print("Combined evidence rows:", len(combined))
print("Pairwise comparisons:", len(pairs))
print("Figure output folder:", OUT)
print("Embedded manuscript:", embedded_manuscript)
print()
print(audit.to_string(index=False))
print()
print("PASS — no ML models retrained; figures generated from audited evidence.")
