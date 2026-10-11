# Phase 12D — Temporal Sample Imbalance Audit

## Summary
           model  total_rows  unique_record_ids  duplicate_record_id_rows  missing_dates first_date  last_date  months_with_records  months_in_full_timeline  months_with_zero_records  minimum_monthly_rows  median_monthly_rows  maximum_monthly_rows  rows_in_largest_month  assignment_coverage_overall
             NMF        4026               4026                         0              0 2014-10-31 2016-02-17                   10                       17                         7                     3                  9.0                  3899                   3899                     0.953304
Semantic K-means        4026               4026                         0              0 2014-10-31 2016-02-17                   10                       17                         7                     3                  9.0                  3899                   3899                     0.962245

## Months with fewer than 30 records
           model   month  corpus_rows  unique_record_ids  assigned_rows  dated_rows  unassigned_rows  assignment_coverage  assigned_share_of_corpus  small_month_under_30_rows
             NMF 2014-10            6                  6              6           6                0             1.000000                  1.000000                       True
             NMF 2014-11           11                 11             11          11                0             1.000000                  1.000000                       True
             NMF 2014-12            4                  4              3           4                1             0.750000                  0.750000                       True
             NMF 2015-01            3                  3              3           3                0             1.000000                  1.000000                       True
             NMF 2015-02            3                  3              2           3                1             0.666667                  0.666667                       True
             NMF 2015-03            7                  7              5           7                2             0.714286                  0.714286                       True
             NMF 2015-04           16                 16             15          16                1             0.937500                  0.937500                       True
Semantic K-means 2014-10            6                  6              6           6                0             1.000000                  1.000000                       True
Semantic K-means 2014-11           11                 11             11          11                0             1.000000                  1.000000                       True
Semantic K-means 2014-12            4                  4              4           4                0             1.000000                  1.000000                       True
Semantic K-means 2015-01            3                  3              3           3                0             1.000000                  1.000000                       True
Semantic K-means 2015-02            3                  3              2           3                1             0.666667                  0.666667                       True
Semantic K-means 2015-03            7                  7              6           7                1             0.857143                  0.857143                       True
Semantic K-means 2015-04           16                 16             15          16                1             0.937500                  0.937500                       True

## Missing months
2015-05, 2015-06, 2015-07, 2015-08, 2015-09, 2015-11, 2015-12

## Interpretation safeguards
- Monthly counts describe the available corpus, not Reddit activity across the entire platform.
- Months with very small samples can produce unstable topic proportions.
- A change in topic share does not by itself establish a change in emotional expression or a causal narrative shift.
- Compare all-corpus shares with shares among assigned documents, and disclose unassigned records.
- The threshold of 30 rows is a flag for review, not a universal adequacy criterion.
- The audit does not change model assignments or original Phase 7O outputs.