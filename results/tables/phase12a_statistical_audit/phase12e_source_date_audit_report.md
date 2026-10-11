# Phase 12E — Source Date and Monthly Distribution Audit

## Source file checks
                         source  status   rows  unique_ids   date_column  missing_dates                first_date                 last_date                                                             detail
        Phase 7J cleaned corpus MISSING    NaN         NaN           NaN            NaN                       NaN                       NaN File not found: results\tables\phase7j_cleaned_documents_audit.csv
    Phase 8 merged observations READ_OK 4874.0      4874.0   created_utc            0.0 2014-10-31 20:59:42+00:00 2016-02-17 05:41:24+00:00                              Columns detected; ID column=record_id
     Phase 7O NMF temporal join READ_OK 4026.0      4026.0 analysis_date            0.0 2014-10-31 00:00:00+00:00 2016-02-17 00:00:00+00:00                              Columns detected; ID column=record_id
Phase 7O Semantic temporal join READ_OK 4026.0      4026.0 analysis_date            0.0 2014-10-31 00:00:00+00:00 2016-02-17 00:00:00+00:00                              Columns detected; ID column=record_id

## Monthly counts by source
  month  Phase 7O NMF temporal join  Phase 7O Semantic temporal join  Phase 8 merged observations
2014-10                           6                                6                            6
2014-11                          11                               11                           11
2014-12                           4                                4                            6
2015-01                           3                                3                            7
2015-02                           3                                3                            4
2015-03                           7                                7                            8
2015-04                          16                               16                           24
2015-10                          40                               40                           47
2016-01                          37                               37                           40
2016-02                        3899                             3899                         4721

## Interpretation
- If the February concentration appears in the Phase 7J source, it predates the Phase 8 and Phase 7O joins.
- If the distribution changes between sources, investigate filtering, joins, date parsing, and duplicate handling.
- This comparison alone does not establish why the source corpus is concentrated in a particular month.
- No source files are changed by this audit.