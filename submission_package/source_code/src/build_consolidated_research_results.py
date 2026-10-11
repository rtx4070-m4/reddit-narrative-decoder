from pathlib import Path
import pandas as pd

root = Path(".")
out = root / "results/tables/phase11_final_report"
out.mkdir(parents=True, exist_ok=True)

phase8 = root / "results/tables/phase8_cross_community"
temporal = root / "results/tables/phase7o_temporal_narratives_v17"
topic = root / "results/tables/phase11a_consolidated_topic_dictionary"

sources = {
    "Community agreement": phase8 / "phase8_vader_transformer_agreement_by_community.csv",
    "Agreement community test": phase8 / "phase8_agreement_community_chisquare.csv",
    "Emotion-label test": phase8 / "phase8_emotion_label_chisquare_all_years.csv",
    "Kruskal-Wallis tests": phase8 / "phase8_kruskal_all_years.csv",
    "Pairwise community tests": phase8 / "phase8_pairwise_all_years.csv",
    "NMF monthly prevalence": temporal / "nmf_monthly_topic_prevalence.csv",
    "Semantic monthly prevalence": temporal / "semantic_kmeans_monthly_topic_prevalence.csv",
    "Consolidated topic dictionary": topic / "phase11a_consolidated_topic_dictionary.csv",
}

report = [
    "# Reddit Narrative Decoder — Consolidated Research Results",
    "",
    "## Scope",
    "This report summarizes existing output tables. It does not rerun models.",
    "",
    "## Source availability",
]
loaded = {}

def add_table(df, limit=30):
    """Render a readable Markdown table without the optional tabulate package."""
    if df.empty:
        return "_No rows available._"
    view = df.head(limit).fillna("")
    columns = [str(c) for c in view.columns]
    lines = [
        "| " + " | ".join(c.replace("|", "\\|") for c in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in view.iterrows():
        values = []
        for value in row.tolist():
            value = str(value).replace("\r", " ").replace("\n", " ").replace("|", "\\|")
            values.append(value)
        lines.append("| " + " | ".join(values) + " |")
    if len(df) > limit:
        lines.append("")
        lines.append(f"_Showing {limit} of {len(df)} rows._")
    return "\n".join(lines)

for name, path in sources.items():
    if not path.exists():
        report.append(f"- MISSING: {name} (`{path.as_posix()}`)")
        continue
    try:
        df = pd.read_csv(path)
        loaded[name] = df
        report.append(f"- {name}: {len(df)} rows, {len(df.columns)} columns.")
    except Exception as exc:
        report.append(f"- READ ERROR: {name}: {exc}")

for name in ["Community agreement", "Agreement community test", "Emotion-label test"]:
    if name in loaded:
        report += ["", f"## {name}", "", add_table(loaded[name]), ""]

for name in ["Kruskal-Wallis tests", "Pairwise community tests"]:
    if name not in loaded:
        continue
    df = loaded[name]
    report += ["", f"## {name}", ""]
    col = "significant_holm_0_05"
    if col in df.columns:
        sig = df[col].astype(str).str.lower().isin(["true", "1", "yes"])
        report.append(f"Rows marked significant after Holm correction: {int(sig.sum())} of {len(df)}.")
        report += ["", "### Rows marked significant", "", add_table(df[sig]), ""]
    else:
        report.append("No recognized Holm-significance column was found; inspect the complete source table.")
        report += ["", add_table(df), ""]

report += ["", "## Temporal topic prevalence", ""]
for name in ["NMF monthly prevalence", "Semantic monthly prevalence"]:
    if name in loaded:
        df = loaded[name]
        months = df["month"].nunique() if "month" in df.columns else "not available"
        report += [
            f"### {name}",
            "",
            f"Rows: {len(df)}; months represented: {months}.",
            "Monthly topic shares are descriptive and do not establish causal change.",
            "",
            add_table(df, limit=30),
            "",
        ]

if "Consolidated topic dictionary" in loaded:
    df = loaded["Consolidated topic dictionary"]
    report += ["", "## Topic dictionary status", "", f"Topic-group records: {len(df)}."]
    if "log_decision" in df.columns:
        counts = df["log_decision"].fillna("missing").astype(str).value_counts()
        report.append("Recorded decisions: " + "; ".join(f"{k}: {v}" for k, v in counts.items()))
    report += [
        "",
        "Groups marked `do_not_label` must not be reported as accepted topic labels.",
        "Batch approval does not constitute independent example-level human validation.",
        "",
    ]

report += [
    "## Limitations",
    "",
    "- Topic labels are interpretations of model-generated clusters, not ground truth.",
    "- VADER and transformer agreement measures agreement, not which method is necessarily correct.",
    "- Statistical significance should be interpreted alongside effect sizes, sample sizes, assumptions, and multiple-testing corrections.",
    "- The temporal dataset covers October 2014 to February 2016; observed periods are limited.",
    "- Structural consistency does not independently establish statistical validity.",
    "",
]

path = out / "consolidated_research_results.md"
path.write_text("\n".join(report), encoding="utf-8")

print("Saved:", path.resolve())
print("Loaded source tables:", len(loaded), "of", len(sources))
print("\n".join(report[:35]))
