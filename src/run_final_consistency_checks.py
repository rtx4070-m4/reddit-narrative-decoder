from pathlib import Path
import pandas as pd
import json

root = Path(".")
base = root / "results/tables/phase7o_temporal_narratives_v17"
out = root / "results/tables/phase11_final_report"
out.mkdir(parents=True, exist_ok=True)

checks = []

def check(name, passed, detail):
    checks.append({
        "check": name,
        "status": "PASS" if passed else "REVIEW",
        "detail": str(detail)
    })

manifest = json.loads((base / "phase7o_manifest.json").read_text(encoding="utf-8"))
audit = pd.read_csv(base / "phase7o_join_audit.csv")
doc = pd.read_csv(base / "nmf_document_topic_date_join.csv")
sem = pd.read_csv(base / "semantic_kmeans_document_topic_date_join.csv")

for model_name, frame in [("NMF", doc), ("Semantic KMeans", sem)]:
    expected = 4026
    unique_ids = frame["record_id"].nunique(dropna=True)
    row_count = len(frame)
    has_assignment = frame["has_assignment"].astype(str).str.lower().isin(
        ["true", "1", "yes"]
    )
    assigned = int(has_assignment.sum())
    unassigned = row_count - assigned

    check(
        f"{model_name}: row count matches corpus",
        row_count == expected,
        f"rows={row_count}; expected={expected}"
    )
    check(
        f"{model_name}: record IDs unique",
        unique_ids == row_count,
        f"rows={row_count}; unique IDs={unique_ids}"
    )
    check(
        f"{model_name}: assigned + unassigned reconcile",
        assigned + unassigned == row_count,
        f"assigned={assigned}; unassigned={unassigned}; rows={row_count}"
    )

for filename, group_cols, total_col, share_col in [
    ("nmf_monthly_topic_prevalence.csv",
     ["month"], "month_total_all_corpus_rows", "share_of_all_corpus_rows"),
    ("semantic_kmeans_monthly_topic_prevalence.csv",
     ["month"], "month_total_all_corpus_rows", "share_of_all_corpus_rows"),
]:
    p = base / filename
    if not p.exists():
        check(filename + ": exists", False, "File missing")
        continue
    df = pd.read_csv(p)
    check(filename + ": has rows", len(df) > 0, f"rows={len(df)}")
    check(filename + ": share values within [0,1]",
          df[share_col].dropna().between(0, 1).all(),
          f"share column={share_col}")

phase8 = root / "results/tables/phase8_cross_community"
summary_files = [
    "phase8_join_audit.csv",
    "phase8_vader_transformer_agreement_by_community.csv",
    "phase8_agreement_community_chisquare.csv",
    "phase8_kruskal_all_years.csv",
    "phase8_pairwise_all_years.csv",
    "phase8_emotion_label_chisquare_all_years.csv",
]
for filename in summary_files:
    p = phase8 / filename
    if p.exists():
        try:
            df = pd.read_csv(p)
            check(f"Phase 8 {filename}: readable",
                  True, f"rows={len(df)}; columns={len(df.columns)}")
        except Exception as exc:
            check(f"Phase 8 {filename}: readable", False, exc)
    else:
        check(f"Phase 8 {filename}: exists", False, "File missing")

result = pd.DataFrame(checks)
result.to_csv(out / "final_consistency_checks.csv", index=False)

print(result.to_string(index=False))
print("\nSaved:", (out / "final_consistency_checks.csv").resolve())
print("\nNote: these are structural consistency checks, not proof of statistical validity.")
