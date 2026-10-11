# Computational Analysis of Emotional Expression, Psychological Language, and Narrative Evolution in Reddit Communities

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
