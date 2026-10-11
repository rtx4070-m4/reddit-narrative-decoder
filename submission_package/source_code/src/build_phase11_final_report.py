from pathlib import Path
import pandas as pd

root = Path(".")
outdir = root / "results" / "tables" / "phase11_final_report"
outdir.mkdir(parents=True, exist_ok=True)

files = {
    "Phase 11A dictionary": root / "results/tables/phase11a_consolidated_topic_dictionary/phase11a_consolidated_topic_dictionary.csv",
    "Phase 11A integration audit": root / "results/tables/phase11a_consolidated_topic_dictionary/phase11a_integration_audit.csv",
    "Phase 11B quality audit": root / "results/tables/phase11b_topic_quality_audit/phase11b_topic_quality_audit.csv",
    "Phase 11B similar label pairs": root / "results/tables/phase11b_topic_quality_audit/phase11b_similar_label_pairs.csv",
    "Phase 11B summary": root / "results/tables/phase11b_topic_quality_audit/phase11b_quality_audit_summary.csv",
    "Phase 11C original comparison": root / "results/tables/phase11c_label_comparison/phase11c_label_evidence_comparison.csv",
    "Phase 11C provisional assessment draft": root / "results/tables/phase11c_label_comparison/phase11c_fast_assessment_draft.csv",
}

lines = [
    "# Reddit Narrative Decoder — Phase 11 Completion Report",
    "",
    "## Scope",
    "This report inventories existing Phase 11 outputs. It does not rerun topic models,",
    "call external APIs, or modify source decision logs.",
    "",
    "## Output inventory",
]

loaded = {}
for name, path in files.items():
    if not path.exists():
        lines.append(f"- **MISSING:** {name} — `{path.as_posix()}`")
        continue
    try:
        df = pd.read_csv(path)
        loaded[name] = df
        lines.append(f"- **Available:** {name} — {len(df)} rows, {len(df.columns)} columns.")
    except Exception as exc:
        lines.append(f"- **READ ERROR:** {name} — {exc}")

lines += ["", "## Findings from available outputs", ""]

if "Phase 11A dictionary" in loaded:
    d = loaded["Phase 11A dictionary"]
    lines.append(f"- Consolidated topic dictionary: **{len(d)} rows**.")
    for col in ["log_decision", "decision", "label_status", "integration_status"]:
        if col in d.columns:
            lines.append(f"- `{col}` counts: " + "; ".join(
                f"{k}={v}" for k, v in d[col].fillna("MISSING").astype(str).value_counts().items()
            ))
            break

if "Phase 11B quality audit" in loaded:
    q = loaded["Phase 11B quality audit"]
    lines.append(f"- Topic quality audit: **{len(q)} rows**.")
    for col in q.columns:
        if "near_duplicate" in col.lower() or "duplicate" in col.lower():
            lines.append(f"- Duplicate-related field `{col}` is available for inspection.")
            break

if "Phase 11B similar label pairs" in loaded:
    p = loaded["Phase 11B similar label pairs"]
    lines.append(f"- Similar-label comparison candidates: **{len(p)} pairs**.")
    lines.append("- Lexical similarity is a screening heuristic, not proof that topics share the same meaning.")

if "Phase 11C provisional assessment draft" in loaded:
    r = loaded["Phase 11C provisional assessment draft"]
    lines.append(f"- Phase 11C draft assessment records: **{len(r)}**.")
    if "log_decision" in r.columns:
        lines.append("- Existing decision counts: " + "; ".join(
            f"{k}={v}" for k, v in r["log_decision"].fillna("MISSING").astype(str).value_counts().items()
        ))
    if "same_underlying_theme" in r.columns:
        lines.append("- Theme-comparison assessments: " + "; ".join(
            f"{k}={v}" for k, v in r["same_underlying_theme"].fillna("blank").astype(str).value_counts().items()
        ))
    lines.append("- These draft assessments are provisional and must not be represented as independently completed human reviews.")

lines += [
    "",
    "## Limitations and interpretation",
    "- Batch approval of recommendations is not equivalent to example-by-example human validation.",
    "- Groups with a recorded `do_not_label` decision must not be presented as accepted topic labels.",
    "- Lexically similar labels require contextual review before merging or renaming.",
    "- Topic labels are interpretations of model-generated clusters, not ground-truth categories.",
    "- This report does not claim causal findings or population-level generalizability.",
    "",
    "## Next steps",
    "1. Inspect the available representative examples for the flagged similar-label pairs.",
    "2. Complete genuine human review where required by the research protocol.",
    "3. Freeze the final dictionary only after documenting label decisions and exclusions.",
    "4. Report the methodology, validation limitations, and reproducibility details.",
    "",
    "## Reproducibility note",
    "Generated from existing CSV outputs. No API calls or topic-model retraining were performed.",
]

report = outdir / "phase11_completion_report.md"
report.write_text("\n".join(lines) + "\n", encoding="utf-8")

inventory = pd.DataFrame([
    {"output": name, "path": path.as_posix(), "exists": path.exists(),
     "rows": len(loaded[name]) if name in loaded else None}
    for name, path in files.items()
])
inventory.to_csv(outdir / "phase11_output_inventory.csv", index=False)

print("REPORT:", report.resolve())
print("INVENTORY:", (outdir / "phase11_output_inventory.csv").resolve())
print("\n".join(lines))
