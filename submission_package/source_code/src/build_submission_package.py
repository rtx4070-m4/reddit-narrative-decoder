from pathlib import Path
import shutil
import zipfile
from datetime import datetime

ROOT = Path(".").resolve()
OUT = ROOT / "submission_package"
ZIP_PATH = ROOT / "reddit_narrative_decoder_submission.zip"

if OUT.exists():
    shutil.rmtree(OUT)

OUT.mkdir(parents=True)

def copy_file(src, dest):
    src = ROOT / src
    if src.exists() and src.is_file():
        target = OUT / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        print("Included:", src.relative_to(ROOT))

# Core reports
core_files = [
    ("results/tables/phase11_final_report/consolidated_research_results.md",
     "reports/consolidated_research_results.md"),
    ("results/tables/phase11_final_report/phase11_completion_report.md",
     "reports/phase11_completion_report.md"),
    ("results/tables/phase11_final_report/completion_audit_report.md",
     "reports/completion_audit_report.md"),
    ("results/tables/phase12a_statistical_audit/phase12d_temporal_audit_report.md",
     "reports/temporal_audit_report.md"),
    ("results/tables/phase12a_statistical_audit/phase12e_source_date_audit_report.md",
     "reports/source_date_audit_report.md"),
]
for src, dest in core_files:
    copy_file(src, dest)

# Key results and audit tables
key_tables = [
    "results/tables/phase8_cross_community/phase8_merged_observations.csv",
    "results/tables/phase8_cross_community/phase8_kruskal_all_years.csv",
    "results/tables/phase8_cross_community/phase8_pairwise_all_years.csv",
    "results/tables/phase8_cross_community/phase8_emotion_label_chisquare_all_years.csv",
    "results/tables/phase11a_consolidated_topic_dictionary/phase11a_consolidated_topic_dictionary.csv",
    "results/tables/phase11a_consolidated_topic_dictionary/phase11a_integration_audit.csv",
    "results/tables/phase11b_topic_quality_audit/phase11b_topic_quality_audit.csv",
    "results/tables/phase11b_topic_quality_audit/phase11b_similar_label_pairs.csv",
    "results/tables/phase11c_label_comparison/phase11c_label_evidence_comparison.csv",
    "results/tables/phase11c_label_comparison/phase11c_fast_assessment_draft.csv",
    "results/tables/phase12a_statistical_audit/phase12a_audit_checks.csv",
    "results/tables/phase12a_statistical_audit/phase12b_kruskal_reproduction.csv",
    "results/tables/phase12a_statistical_audit/phase12c_pairwise_reproduction.csv",
    "results/tables/phase12a_statistical_audit/phase12d_monthly_coverage.csv",
    "results/tables/phase12a_statistical_audit/phase12d_temporal_summary.csv",
    "results/tables/phase12a_statistical_audit/phase12e_source_monthly_comparison.csv",
    "results/tables/phase12a_statistical_audit/phase12f_record_id_membership.csv",
    "results/tables/phase12a_statistical_audit/phase12g_extra_records_month_community.csv",
    "results/tables/phase12a_statistical_audit/phase12h_linguistic_ids_absent_from_phase7j.csv",
    "results/tables/phase12a_statistical_audit/phase12j_processed_corpus_extra_ids.csv",
    "results/tables/phase12a_statistical_audit/phase12n_nlp_ids_absent_from_phase7h.csv",
    "results/tables/phase11_final_report/project_file_inventory.csv",
]
for src in key_tables:
    copy_file(src, "results/" + Path(src).name)

# Include source code only, never secrets, environments, or raw data.
src_root = ROOT / "src"
if src_root.exists():
    for src in src_root.rglob("*.py"):
        if any(part in {".venv", "__pycache__"} for part in src.parts):
            continue
        rel = src.relative_to(ROOT)
        target = OUT / "source_code" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)

report = r"""# Computational Analysis of Emotional Expression, Psychological Language, and Narrative Evolution in Reddit Communities

## Project overview

This project analyzes emotional expression, psychological-language signals, topic structure, and cross-community differences in Reddit discussions. The analyses combine sentiment and emotion outputs, linguistic features, topic modeling, cross-community statistical comparisons, and a temporal-topic audit.

## Data and analytical coverage

The Phase 8 merged analytical dataset contains 4,874 observations across gaming, science, todayilearned, and worldnews. The temporal topic-modeling dataset contains 4,026 records. These datasets do not have identical record coverage.

## Main findings

- VADER–transformer emotion agreement varies across communities. Agreement is not equivalent to accuracy, and neither method is treated as ground truth.
- The community association test for VADER/transformer agreement reported chi-square = 11.646, df = 3, p = 0.008699, Cramér's V = 0.04888.
- The emotion-label distribution test reported chi-square = 87.778, df = 18, p = 3.609e-11, Cramér's V = 0.07748.
- The saved statistical audit independently reproduced 25 Kruskal–Wallis tests and 150 pairwise comparisons. Statistical significance should be interpreted alongside the generally small effect sizes.
- The consolidated topic dictionary contains 40 topic groups. Five groups remain marked do_not_label; the batch approval process is not equivalent to independent example-level human validation.
- The temporal topic dataset spans October 2014 to February 2016, but 3,899 of its 4,026 records (96.85%) are in February 2016. The data therefore do not support a strong claim of continuous month-to-month narrative evolution.

## Methodological and data-quality limitations

1. The 4,874-record Phase 8 corpus and the 4,026-record topic-modeling corpus differ by 848 record IDs. The discrepancy is documented but unresolved in this submission.
2. The master corpus has 4,972 rows; 98 records are absent from the processed NLP corpus. This is a separate coverage difference.
3. VADER and transformer outputs are computational measurements, not verified psychological diagnoses or ground-truth emotional states.
4. Topic labels are interpretive summaries, not definitive ground-truth categories. Some topic groups remain unlabelled.
5. Temporal coverage is extremely uneven, with nearly all temporal-topic records in one month. Claims about narrative evolution over time must remain limited.
6. Statistical associations do not establish causality.
7. Automated statistical reproduction verifies calculations against the saved inputs; it does not independently validate the dataset, measurement assumptions, or substantive interpretation.

## Reproducibility and included materials

The package includes available reports, selected analytical tables and audit outputs, and Python source scripts. Raw input data and environment files are intentionally not bundled. Paths and file availability can differ between installations.

## Submission note

This report is a summary of the existing saved results, not a claim that every project component has been independently validated. Review the included reports and tables before presenting numerical findings.
"""

(OUT / "FINAL_PROJECT_REPORT.md").write_text(report, encoding="utf-8")

readme = """# Submission Package

Start with `reports/` and `FINAL_PROJECT_REPORT.md`.

- `reports/`: previously generated research and audit reports.
- `results/`: selected analytical tables, topic-label audits, statistical reproduction results, and data-coverage checks.
- `source_code/`: Python source scripts copied from `src/`.

Important:
- Phase 8 contains 4,874 observations; the topic-modeling dataset contains 4,026.
- The 848-record discrepancy remains unresolved and is disclosed in the report.
- The temporal topic dataset is heavily concentrated in February 2016.
- Human topic-label review is provisional where indicated.
- `.env`, credentials, `.venv`, raw data files, and caches are not included.

The package preserves existing outputs and does not refit models.
"""
(OUT / "README.md").write_text(readme, encoding="utf-8")

# Inventory the package before zipping it.
inventory = []
for p in sorted(OUT.rglob("*")):
    if p.is_file():
        inventory.append(f"{p.relative_to(OUT)}\t{p.stat().st_size} bytes")
(OUT / "PACKAGE_INVENTORY.txt").write_text(
    "\n".join(inventory) + "\n", encoding="utf-8"
)

if ZIP_PATH.exists():
    ZIP_PATH.unlink()

with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            zf.write(p, p.relative_to(ROOT))

print("\n=== PACKAGE SUMMARY ===")
print("Folder:", OUT)
print("ZIP:", ZIP_PATH)
print("Files included:", sum(1 for p in OUT.rglob("*") if p.is_file()))
print("ZIP size (MB):", round(ZIP_PATH.stat().st_size / (1024 * 1024), 2))
print("Generated:", datetime.now().isoformat(timespec="seconds"))
print("\nCheck any 'Included' omissions above: missing source files were skipped.")
