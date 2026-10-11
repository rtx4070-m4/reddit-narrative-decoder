from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(".")
PHASE8 = ROOT / "results/tables/phase8_cross_community"
TEMPORAL = ROOT / "results/tables/phase7o_temporal_narratives_v17"
OUT = ROOT / "results/tables/phase12a_statistical_audit"
OUT.mkdir(parents=True, exist_ok=True)

checks = []

def record(area, check, status, detail):
    checks.append({
        "area": area,
        "check": check,
        "status": status,
        "detail": str(detail)
    })

def read_csv(path, area):
    if not path.exists():
        record(area, "Source file exists", "REVIEW", f"Missing: {path}")
        return None
    try:
        df = pd.read_csv(path)
        record(area, "Source file readable", "PASS", f"{len(df)} rows, {len(df.columns)} columns")
        return df
    except Exception as exc:
        record(area, "Source file readable", "FAIL", repr(exc))
        return None

# 1. Kruskal-Wallis result and correction audit
kw_path = PHASE8 / "phase8_kruskal_all_years.csv"
kw = read_csv(kw_path, "Kruskal-Wallis")

if kw is not None:
    required = ["p_raw", "p_holm_across_features", "epsilon_squared", "feature"]
    missing = [c for c in required if c not in kw.columns]

    if missing:
        record("Kruskal-Wallis", "Required columns", "REVIEW", f"Missing columns: {missing}")
    else:
        p = pd.to_numeric(kw["p_raw"], errors="coerce")
        ph = pd.to_numeric(kw["p_holm_across_features"], errors="coerce")
        effect = pd.to_numeric(kw["epsilon_squared"], errors="coerce")

        record("Kruskal-Wallis", "Raw p-values valid", 
               "PASS" if p.notna().all() and p.between(0, 1).all() else "REVIEW",
               f"valid={int((p.notna() & p.between(0,1)).sum())}/{len(p)}")

        record("Kruskal-Wallis", "Holm p-values valid",
               "PASS" if ph.notna().all() and ph.between(0, 1).all() else "REVIEW",
               f"valid={int((ph.notna() & ph.between(0,1)).sum())}/{len(ph)}")

        record("Kruskal-Wallis", "Epsilon-squared range",
               "PASS" if effect.notna().all() and effect.between(0, 1).all() else "REVIEW",
               f"min={effect.min()}; max={effect.max()}")

        if "significant_holm_0_05" in kw.columns:
            saved = kw["significant_holm_0_05"].astype(str).str.lower().isin(["true", "1", "yes"])
            recomputed = ph < 0.05
            mismatches = int((saved != recomputed).sum())
            record("Kruskal-Wallis", "Significance flags match adjusted p-values",
                   "PASS" if mismatches == 0 else "REVIEW",
                   f"mismatches={mismatches}; saved significant={int(saved.sum())}; recomputed={int(recomputed.sum())}")

        kw_audit = kw.copy()
        kw_audit["audit_p_raw"] = p
        kw_audit["audit_p_holm"] = ph
        kw_audit["audit_epsilon_squared"] = effect
        if "significant_holm_0_05" in kw.columns:
            kw_audit["audit_significance_recomputed"] = ph < 0.05
        kw_audit.to_csv(OUT / "phase12a_kruskal_review.csv", index=False)

# 2. Pairwise tests and within-feature correction
pw_path = PHASE8 / "phase8_pairwise_all_years.csv"
pw = read_csv(pw_path, "Pairwise tests")

if pw is not None:
    required = ["p_raw", "p_holm_within_feature", "feature"]
    missing = [c for c in required if c not in pw.columns]

    if missing:
        record("Pairwise tests", "Required columns", "REVIEW", f"Missing columns: {missing}")
    else:
        p = pd.to_numeric(pw["p_raw"], errors="coerce")
        ph = pd.to_numeric(pw["p_holm_within_feature"], errors="coerce")
        record("Pairwise tests", "Raw and adjusted p-values in range",
               "PASS" if p.notna().all() and ph.notna().all() and p.between(0,1).all() and ph.between(0,1).all() else "REVIEW",
               f"rows={len(p)}")

        # Independently recompute Holm adjustment within each feature.
        def holm_adjust(values):
            values = np.asarray(values, dtype=float)
            n = len(values)
            order = np.argsort(values)
            sorted_p = values[order]
            adjusted_sorted = np.maximum.accumulate(
                np.minimum(1.0, (n - np.arange(n)) * sorted_p)
            )
            adjusted = np.empty(n)
            adjusted[order] = adjusted_sorted
            return adjusted

        recomputed = pd.Series(np.nan, index=pw.index, dtype=float)
        valid_all = True
        for _, indices in pw.groupby("feature", dropna=False).groups.items():
            idx = list(indices)
            vals = p.loc[idx]
            if vals.isna().any():
                valid_all = False
                continue
            recomputed.loc[idx] = holm_adjust(vals.to_numpy())

        diffs = (recomputed - ph).abs()
        mismatch_count = int((diffs > 1e-8).sum())
        if not valid_all:
            status = "REVIEW"
        else:
            status = "PASS" if mismatch_count == 0 else "REVIEW"
        record("Pairwise tests", "Holm corrections independently reproduced",
               status, f"mismatches beyond 1e-8={mismatch_count}; rows={len(pw)}")

        if "significant_holm_0_05" in pw.columns:
            saved = pw["significant_holm_0_05"].astype(str).str.lower().isin(["true", "1", "yes"])
            mismatch = int((saved != (ph < 0.05)).sum())
            record("Pairwise tests", "Significance flags match adjusted p-values",
                   "PASS" if mismatch == 0 else "REVIEW",
                   f"mismatches={mismatch}; saved significant={int(saved.sum())}")

# 3. Temporal prevalence audit
for filename in [
    "nmf_monthly_topic_prevalence.csv",
    "semantic_kmeans_monthly_topic_prevalence.csv"
]:
    df = read_csv(TEMPORAL / filename, filename)
    if df is None:
        continue

    needed = ["month", "document_count", "month_total_all_corpus_rows",
              "share_of_all_corpus_rows"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        record(filename, "Required columns", "REVIEW", f"Missing: {missing}")
        continue

    count = pd.to_numeric(df["document_count"], errors="coerce")
    total = pd.to_numeric(df["month_total_all_corpus_rows"], errors="coerce")
    share = pd.to_numeric(df["share_of_all_corpus_rows"], errors="coerce")

    valid = count.notna() & total.notna() & (total > 0)
    expected = count[valid] / total[valid]
    difference = (share[valid] - expected).abs()

    record(filename, "Shares equal document count / all-corpus monthly total",
           "PASS" if len(difference) > 0 and difference.max() < 1e-8 else "REVIEW",
           f"checked={len(difference)}; maximum absolute difference={difference.max() if len(difference) else 'NA'}")

    month_totals = df.groupby("month")["month_total_all_corpus_rows"].nunique()
    inconsistent = int((month_totals > 1).sum())
    record(filename, "One all-corpus denominator per month",
           "PASS" if inconsistent == 0 else "REVIEW",
           f"months with inconsistent totals={inconsistent}")

    monthly = df.groupby("month", as_index=False).agg(
        all_corpus_rows=("month_total_all_corpus_rows", "max"),
        topic_rows=("document_count", "sum")
    )
    monthly["difference_rows"] = monthly["all_corpus_rows"] - monthly["topic_rows"]
    monthly.to_csv(OUT / (Path(filename).stem + "_monthly_totals_audit.csv"), index=False)

# 4. Save audit results
audit = pd.DataFrame(checks)
audit.to_csv(OUT / "phase12a_audit_checks.csv", index=False)

print("\n=== PHASE 12A AUDIT CHECKS ===")
print(audit.to_string(index=False))
print("\nSaved folder:", OUT.resolve())
print("\nImportant: a PASS confirms only the specific arithmetic or structural check, not the scientific validity of the study.")
