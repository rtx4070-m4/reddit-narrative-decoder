# Project Completion Audit

Read-only structural audit of existing files. File presence does not establish analytical validity.

- Files under results: 377
- CSV files inspected: 354
- Empty files: 1
- CSV read errors: 1
- Filenames occurring more than once: 6

## Result folders
- results\figures
- results\tables
- results\tables\phase10b_topic_quality_audit
- results\tables\phase10c_topic_assignment_audit
- results\tables\phase10d_denominator_corrected_prevalence
- results\tables\phase10d_monthly_prevalence_corrected
- results\tables\phase10e_february_cross_community
- results\tables\phase10f_topic_distribution_tests
- results\tables\phase10g_topic_interpretability_audit
- results\tables\phase10h_topic_review
- results\tables\phase10i_topic_dictionary
- results\tables\phase10k_expanded_topic_validation
- results\tables\phase10l_coherence_diagnostics
- results\tables\phase10m_semantic_coherence
- results\tables\phase10n_combined_topic_quality
- results\tables\phase10o_a_ai_review_draft
- results\tables\phase10o_human_validation
- results\tables\phase10p_review_pack
- results\tables\phase10r_expanded_evidence
- results\tables\phase10s_label_revision
- results\tables\phase11_final_report
- results\tables\phase11a_consolidated_topic_dictionary
- results\tables\phase11b_topic_quality_audit
- results\tables\phase11c_label_comparison
- results\tables\phase7c_nmf
- results\tables\phase7d_nmf
- results\tables\phase7e_nmf
- results\tables\phase7f_nmf
- results\tables\phase7g_nmf
- results\tables\phase7h_nmf
- results\tables\phase7i_nmf
- results\tables\phase7j_nmf
- results\tables\phase7k_nmf
- results\tables\phase7l_semantic
- results\tables\phase7m_hdbscan
- results\tables\phase7n_hdbscan_sensitivity
- results\tables\phase7o_temporal_narratives
- results\tables\phase7o_temporal_narratives_v17
- results\tables\phase8_cross_community
- results\tables\phase8_cross_community\narrative_evolution_audit

## Interpretation cautions
- Duplicate filenames can be legitimate when stored in different folders; inspect paths before treating them as conflicts.
- CSV row counts are approximate physical line counts and may be inaccurate when quoted fields contain newlines.
- Structural checks do not verify statistical assumptions, sample provenance, temporal leakage, or reproducibility.
- Phase 11C assessments are provisional drafts, not completed independent human validation.
- Preserve all existing source data and decisions until discrepancies are investigated.

## Files generated
- completion_file_inventory.csv
- completion_csv_audit.csv
- completion_duplicate_filenames.csv
