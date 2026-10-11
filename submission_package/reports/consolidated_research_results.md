# Reddit Narrative Decoder — Consolidated Research Results

## Scope
This report summarizes existing output tables. It does not rerun models.

## Source availability
- Community agreement: 4 rows, 9 columns.
- Agreement community test: 1 rows, 7 columns.
- Emotion-label test: 1 rows, 6 columns.
- Kruskal-Wallis tests: 25 rows, 10 columns.
- Pairwise community tests: 150 rows, 15 columns.
- NMF monthly prevalence: 36 rows, 8 columns.
- Semantic monthly prevalence: 30 rows, 8 columns.
- Consolidated topic dictionary: 40 rows, 62 columns.

## Community agreement

| community | n | agreement_count | observed_agreement | observed_agreement_percent | cohen_kappa | vader_neutral_n | transformer_neutral_n | confusion_matrix_file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gaming | 1684 | 1193 | 0.7084323040380047 | 70.84323040380048 | 0.3458450554910331 | 484 | 627 | phase8_agreement_confusion_gaming.csv |
| science | 532 | 336 | 0.631578947368421 | 63.1578947368421 | 0.190874524714829 | 119 | 223 | phase8_agreement_confusion_science.csv |
| todayilearned | 1373 | 938 | 0.6831755280407866 | 68.31755280407866 | 0.301311499573596 | 395 | 536 | phase8_agreement_confusion_todayilearned.csv |
| worldnews | 1285 | 871 | 0.6778210116731518 | 67.78210116731518 | 0.248654756020055 | 329 | 455 | phase8_agreement_confusion_worldnews.csv |


## Agreement community test

| test | n | chi_square | degrees_of_freedom | p_value | cramers_v | interpretation_note |
| --- | --- | --- | --- | --- | --- | --- |
| Pearson chi-square: agreement vs disagreement by community | 4874 | 11.646177861300004 | 3 | 0.0086989177915528 | 0.0488819977580857 | Tests whether observed binary agreement rates vary by community; does not establish classifier accuracy. |


## Emotion-label test

| period | n | chi2 | df | p_value | cramers_v |
| --- | --- | --- | --- | --- | --- |
| all_years | 4874 | 87.77812871431811 | 18 | 3.6092859441092075e-11 | 0.0774800280540214 |


## Kruskal-Wallis tests

Rows marked significant after Holm correction: 22 of 25.

### Rows marked significant

| period | feature | n | groups | kruskal_h | df | p_raw | epsilon_squared | p_holm_across_features | significant_holm_0_05 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_years | word_count | 4874 | 4 | 217.5702393283663 | 3 | 6.728916694340985e-47 | 0.0440595973980218 | 1.6822291735852462e-45 | True |
| all_years | lexical_diversity | 4874 | 4 | 122.61238471596576 | 3 | 2.1123498495640446e-26 | 0.0245610646234015 | 4.858404653997304e-25 | True |
| all_years | first_person_ratio | 4874 | 4 | 40.90616006123298 | 3 | 6.845843933582794e-09 | 0.0077836057620601 | 8.899597113657633e-08 | True |
| all_years | second_person_ratio | 4874 | 4 | 125.88389664333086 | 3 | 4.168636978158388e-27 | 0.0252328329863102 | 1.0004728747580131e-25 | True |
| all_years | third_person_ratio | 4874 | 4 | 24.413834968246924 | 3 | 2.047232094203428e-05 | 0.0043970913692498 | 0.0001433062465942 | True |
| all_years | negation_ratio | 4874 | 4 | 32.322070510977404 | 3 | 4.47613585277986e-07 | 0.0060209590371616 | 4.923749438057846e-06 | True |
| all_years | cognitive_ratio | 4874 | 4 | 52.23662065471889 | 3 | 2.6667172246375764e-11 | 0.0101101890461435 | 4.0000758369563646e-10 | True |
| all_years | uncertainty_ratio | 4874 | 4 | 93.87863993172569 | 3 | 3.2158085966882014e-20 | 0.0186609116902927 | 6.753198053045222e-19 | True |
| all_years | social_ratio | 4874 | 4 | 83.54236343186501 | 3 | 5.33344643444223e-18 | 0.0165384729839558 | 1.066689286888446e-16 | True |
| all_years | achievement_ratio | 4874 | 4 | 38.89090490923525 | 3 | 1.8305282534232774e-08 | 0.0073697956692474 | 2.196633904107933e-07 | True |
| all_years | risk_threat_ratio | 4874 | 4 | 26.442167564343222 | 3 | 7.705929039497881e-06 | 0.0048135867688589 | 6.164743231598305e-05 | True |
| all_years | hedge_ratio | 4874 | 4 | 60.54119890340388 | 3 | 4.5041720162511734e-13 | 0.0118154412532656 | 7.657092427626993e-12 | True |
| all_years | question_word_ratio | 4874 | 4 | 77.59460372919276 | 3 | 1.0066866039452344e-16 | 0.0153171670901833 | 1.9127045474959453e-15 | True |
| all_years | vader_compound | 4874 | 4 | 54.44388531567423 | 3 | 9.022836027293788e-12 | 0.0105634261428489 | 1.4436537643670058e-10 | True |
| all_years | vader_negative | 4874 | 4 | 21.45924525682555 | 3 | 8.451109911344069e-05 | 0.0037903994367198 | 0.0005070665946806 | True |
| all_years | vader_neutral | 4874 | 4 | 27.89796769143944 | 3 | 3.815618402515179e-06 | 0.0051125190331497 | 3.434056562263661e-05 | True |
| all_years | emotion_prob_anger | 4874 | 4 | 74.82639529763705 | 3 | 3.947426951184835e-16 | 0.0147487464676872 | 7.105368512132704e-15 | True |
| all_years | emotion_prob_disgust | 4874 | 4 | 31.34403783420957 | 3 | 7.194366108579359e-07 | 0.0058201309721169 | 7.194366108579359e-06 | True |
| all_years | emotion_prob_fear | 4874 | 4 | 96.53220454395228 | 3 | 8.649785629499472e-21 | 0.0192057914874645 | 1.9029528384898842e-19 | True |
| all_years | emotion_prob_joy | 4874 | 4 | 43.07491663197972 | 3 | 2.3724964060506834e-09 | 0.0082289356533839 | 3.3214949684709575e-08 | True |
| all_years | emotion_prob_sadness | 4874 | 4 | 11.64952359689479 | 3 | 0.0086854521833661 | 0.0017760828741057 | 0.0347418087334644 | True |
| all_years | emotion_prob_surprise | 4874 | 4 | 17.056920781833263 | 3 | 0.0006879449696225 | 0.0028864313720396 | 0.0034397248481125 | True |


## Pairwise community tests

Rows marked significant after Holm correction: 91 of 150.

### Rows marked significant

| period | feature | community_a | community_b | n_a | n_b | median_a | median_b | mean_a | mean_b | mannwhitney_u | p_raw | rank_biserial_a_vs_b | p_holm_within_feature | significant_holm_0_05 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_years | word_count | gaming | science | 1684 | 532 | 13.0 | 28.0 | 21.144299287410927 | 51.2593984962406 | 259879.0 | 1.9890752602755785e-48 | -0.4198404264818817 | 1.1934451561653472e-47 | True |
| all_years | word_count | gaming | todayilearned | 1684 | 1373 | 13.0 | 15.0 | 21.144299287410927 | 27.983976693372178 | 1042294.5 | 2.741691024688579e-06 | -0.0984126338807646 | 5.483382049377158e-06 | True |
| all_years | word_count | gaming | worldnews | 1684 | 1285 | 13.0 | 15.0 | 21.144299287410927 | 28.470817120622566 | 943403.0 | 2.0949277501868155e-09 | -0.1280691701248648 | 6.284783250560446e-09 | True |
| all_years | word_count | science | todayilearned | 532 | 1373 | 28.0 | 15.0 | 51.2593984962406 | 27.983976693372178 | 482577.0 | 1.1676570502211368e-27 | 0.3213395834816466 | 5.838285251105684e-27 | True |
| all_years | word_count | science | worldnews | 532 | 1285 | 28.0 | 15.0 | 51.2593984962406 | 28.470817120622566 | 443532.0 | 1.5510682185739852e-23 | 0.2975980808051257 | 6.204272874295941e-23 | True |
| all_years | lexical_diversity | gaming | science | 1684 | 532 | 1.0 | 0.8918918918918919 | 0.9268458213326972 | 0.8707358574236613 | 581962.0 | 3.923403064520266e-28 | 0.2991847195185113 | 2.35404183871216e-27 | True |
| all_years | lexical_diversity | gaming | todayilearned | 1684 | 1373 | 1.0 | 0.9583333333333334 | 0.9268458213326972 | 0.9189614538052728 | 1214392.5 | 0.0099317365597866 | 0.0504525693169766 | 0.0198634731195732 | True |
| all_years | lexical_diversity | gaming | worldnews | 1684 | 1285 | 1.0 | 0.9565217391304348 | 0.9268458213326972 | 0.9188526556420844 | 1144387.5 | 0.0038904520025372 | 0.057688752922909 | 0.0116713560076116 | True |
| all_years | lexical_diversity | science | todayilearned | 532 | 1373 | 0.8918918918918919 | 0.9583333333333334 | 0.8707358574236613 | 0.9189614538052728 | 274880.0 | 2.2145724118581865e-18 | -0.2473536353629887 | 1.1072862059290933e-17 | True |
| all_years | lexical_diversity | science | worldnews | 532 | 1285 | 0.8918918918918919 | 0.9565217391304348 | 0.8707358574236613 | 0.9188526556420844 | 259299.0 | 3.666684152333607e-17 | -0.2413943418858429 | 1.4666736609334426e-16 | True |
| all_years | first_person_ratio | gaming | todayilearned | 1684 | 1373 | 0.0 | 0.0 | 0.0423021541494411 | 0.0336391749231074 | 1231621.0 | 0.0005506641048153 | 0.0653552651838216 | 0.0022026564192612 | True |
| all_years | first_person_ratio | gaming | worldnews | 1684 | 1285 | 0.0 | 0.0 | 0.0423021541494411 | 0.0271097401692669 | 1197335.5 | 2.6422191954160243e-08 | 0.1066254147527194 | 1.5853315172496144e-07 | True |
| all_years | first_person_ratio | science | todayilearned | 532 | 1373 | 0.0123758594346829 | 0.0 | 0.0327248741617157 | 0.0336391749231074 | 394967.5 | 0.0024272794274549 | 0.0814568285243333 | 0.0072818382823649 | True |
| all_years | first_person_ratio | science | worldnews | 532 | 1285 | 0.0123758594346829 | 0.0 | 0.0327248741617157 | 0.0271097401692669 | 387012.5 | 8.891366203680102e-07 | 0.1322445218103625 | 4.445683101840051e-06 | True |
| all_years | second_person_ratio | gaming | todayilearned | 1684 | 1373 | 0.0 | 0.0 | 0.0199599327288576 | 0.011477628722398 | 1262765.5 | 4.88255462351292e-10 | 0.092295336079428 | 1.953021849405168e-09 | True |
| all_years | second_person_ratio | gaming | worldnews | 1684 | 1285 | 0.0 | 0.0 | 0.0199599327288576 | 0.0074777841403733 | 1230603.5 | 3.591063983593572e-21 | 0.1373730325240072 | 2.1546383901561436e-20 | True |
| all_years | second_person_ratio | science | todayilearned | 532 | 1373 | 0.0 | 0.0 | 0.0129782141564681 | 0.011477628722398 | 410044.5 | 1.883733213012686e-09 | 0.1227390216254402 | 5.651199639038059e-09 | True |
| all_years | second_person_ratio | science | worldnews | 532 | 1285 | 0.0 | 0.0 | 0.0129782141564681 | 0.0074777841403733 | 400993.0 | 2.1220157002599888e-19 | 0.1731458997688775 | 1.0610078501299944e-18 | True |
| all_years | second_person_ratio | todayilearned | worldnews | 1373 | 1285 | 0.0 | 0.0 | 0.011477628722398 | 0.0074777841403733 | 923362.0 | 0.0004581755381181 | 0.0467147120254152 | 0.0009163510762362 | True |
| all_years | third_person_ratio | gaming | todayilearned | 1684 | 1373 | 0.0 | 0.0 | 0.0311481843235812 | 0.0378572368972333 | 1058236.5 | 7.168431230732431e-06 | -0.0846227637522425 | 4.3010587384394586e-05 | True |
| all_years | third_person_ratio | gaming | worldnews | 1684 | 1285 | 0.0 | 0.0 | 0.0311481843235812 | 0.0361218324237064 | 1008064.0 | 0.0003639567959918 | -0.0683068846640849 | 0.0018197839799593 | True |
| all_years | negation_ratio | gaming | science | 1684 | 532 | 0.0 | 0.0 | 0.0100593808644755 | 0.009134286282139 | 403292.5 | 1.969346525260896e-07 | -0.0996809869090779 | 1.1816079151565373e-06 | True |
| all_years | negation_ratio | gaming | todayilearned | 1684 | 1373 | 0.0 | 0.0 | 0.0100593808644755 | 0.0088284210606963 | 1115888.5 | 0.0112520367160554 | -0.0347536386330884 | 0.0337561101481664 | True |
| all_years | negation_ratio | gaming | worldnews | 1684 | 1285 | 0.0 | 0.0 | 0.0100593808644755 | 0.0125456648256835 | 1017378.0 | 2.902084620331484e-05 | -0.0596985128977698 | 0.0001451042310165 | True |
| all_years | negation_ratio | science | todayilearned | 532 | 1373 | 0.0 | 0.0 | 0.009134286282139 | 0.0088284210606963 | 389386.5 | 0.0016932196789797 | 0.0661755444693306 | 0.0067728787159191 | True |
| all_years | cognitive_ratio | gaming | science | 1684 | 532 | 0.0 | 0.0 | 0.0073365089072768 | 0.0094105359109059 | 390247.0 | 1.2775862306290314e-11 | -0.1288040469344382 | 7.665517383774188e-11 | True |
| all_years | cognitive_ratio | gaming | todayilearned | 1684 | 1373 | 0.0 | 0.0 | 0.0073365089072768 | 0.0098180518230193 | 1107753.0 | 0.0018146730528165 | -0.0417908666114218 | 0.0054440191584495 | True |
| all_years | cognitive_ratio | science | todayilearned | 532 | 1373 | 0.0 | 0.0 | 0.0094105359109059 | 0.0098180518230193 | 396195.5 | 5.632986455838371e-05 | 0.0848192038727555 | 0.0002253194582335 | True |
| all_years | cognitive_ratio | science | worldnews | 532 | 1285 | 0.0 | 0.0 | 0.0094105359109059 | 0.007307622355636 | 383528.5 | 2.0954123728435755e-09 | 0.1220517246423451 | 1.0477061864217876e-08 | True |
| all_years | cognitive_ratio | todayilearned | worldnews | 1373 | 1285 | 0.0 | 0.0 | 0.0098180518230193 | 0.007307622355636 | 912353.5 | 0.0189402060512246 | 0.0342355771819498 | 0.0378804121024492 | True |

_Showing 30 of 91 rows._


## Temporal topic prevalence

### NMF monthly prevalence

Rows: 36; months represented: 10.
Monthly topic shares are descriptive and do not establish causal change.

| model | month | topic_label | document_count | month_total_all_corpus_rows | month_total_assigned_rows | share_of_all_corpus_rows | share_among_assigned_rows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NMF | 2014-10 | Topic 1 | 2 | 6 | 6 | 0.3333333333333333 | 0.3333333333333333 |
| NMF | 2014-10 | Topic 3 | 1 | 6 | 6 | 0.1666666666666666 | 0.1666666666666666 |
| NMF | 2014-10 | Topic 4 | 3 | 6 | 6 | 0.5 | 0.5 |
| NMF | 2014-11 | Topic 1 | 6 | 11 | 11 | 0.5454545454545454 | 0.5454545454545454 |
| NMF | 2014-11 | Topic 2 | 1 | 11 | 11 | 0.0909090909090909 | 0.0909090909090909 |
| NMF | 2014-11 | Topic 3 | 1 | 11 | 11 | 0.0909090909090909 | 0.0909090909090909 |
| NMF | 2014-11 | Topic 4 | 3 | 11 | 11 | 0.2727272727272727 | 0.2727272727272727 |
| NMF | 2014-12 | Topic 1 | 3 | 4 | 3 | 0.75 | 1.0 |
| NMF | 2014-12 | UNASSIGNED | 1 | 4 | 3 | 0.25 |  |
| NMF | 2015-01 | Topic 1 | 1 | 3 | 3 | 0.3333333333333333 | 0.3333333333333333 |
| NMF | 2015-01 | Topic 3 | 2 | 3 | 3 | 0.6666666666666666 | 0.6666666666666666 |
| NMF | 2015-02 | Topic 5 | 2 | 3 | 2 | 0.6666666666666666 | 1.0 |
| NMF | 2015-02 | UNASSIGNED | 1 | 3 | 2 | 0.3333333333333333 |  |
| NMF | 2015-03 | Topic 1 | 2 | 7 | 5 | 0.2857142857142857 | 0.4 |
| NMF | 2015-03 | Topic 3 | 2 | 7 | 5 | 0.2857142857142857 | 0.4 |
| NMF | 2015-03 | Topic 4 | 1 | 7 | 5 | 0.1428571428571428 | 0.2 |
| NMF | 2015-03 | UNASSIGNED | 2 | 7 | 5 | 0.2857142857142857 |  |
| NMF | 2015-04 | Topic 1 | 5 | 16 | 15 | 0.3125 | 0.3333333333333333 |
| NMF | 2015-04 | Topic 3 | 2 | 16 | 15 | 0.125 | 0.1333333333333333 |
| NMF | 2015-04 | Topic 4 | 4 | 16 | 15 | 0.25 | 0.2666666666666666 |
| NMF | 2015-04 | Topic 5 | 4 | 16 | 15 | 0.25 | 0.2666666666666666 |
| NMF | 2015-04 | UNASSIGNED | 1 | 16 | 15 | 0.0625 |  |
| NMF | 2015-10 | Topic 1 | 20 | 40 | 38 | 0.5 | 0.5263157894736842 |
| NMF | 2015-10 | Topic 3 | 9 | 40 | 38 | 0.225 | 0.2368421052631578 |
| NMF | 2015-10 | Topic 4 | 3 | 40 | 38 | 0.075 | 0.0789473684210526 |
| NMF | 2015-10 | Topic 5 | 6 | 40 | 38 | 0.15 | 0.1578947368421052 |
| NMF | 2015-10 | UNASSIGNED | 2 | 40 | 38 | 0.05 |  |
| NMF | 2016-01 | Topic 1 | 20 | 37 | 37 | 0.5405405405405406 | 0.5405405405405406 |
| NMF | 2016-01 | Topic 3 | 4 | 37 | 37 | 0.1081081081081081 | 0.1081081081081081 |
| NMF | 2016-01 | Topic 5 | 13 | 37 | 37 | 0.3513513513513513 | 0.3513513513513513 |

_Showing 30 of 36 rows._

### Semantic monthly prevalence

Rows: 30; months represented: 10.
Monthly topic shares are descriptive and do not establish causal change.

| model | month | topic_label | document_count | month_total_all_corpus_rows | month_total_assigned_rows | share_of_all_corpus_rows | share_among_assigned_rows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Semantic_KMeans | 2014-10 | Topic 1 | 6 | 6 | 6 | 1.0 | 1.0 |
| Semantic_KMeans | 2014-11 | Topic 1 | 9 | 11 | 11 | 0.8181818181818182 | 0.8181818181818182 |
| Semantic_KMeans | 2014-11 | Topic 5 | 2 | 11 | 11 | 0.1818181818181818 | 0.1818181818181818 |
| Semantic_KMeans | 2014-12 | Topic 5 | 4 | 4 | 4 | 1.0 | 1.0 |
| Semantic_KMeans | 2015-01 | Topic 1 | 1 | 3 | 3 | 0.3333333333333333 | 0.3333333333333333 |
| Semantic_KMeans | 2015-01 | Topic 4 | 1 | 3 | 3 | 0.3333333333333333 | 0.3333333333333333 |
| Semantic_KMeans | 2015-01 | Topic 5 | 1 | 3 | 3 | 0.3333333333333333 | 0.3333333333333333 |
| Semantic_KMeans | 2015-02 | Topic 2 | 1 | 3 | 2 | 0.3333333333333333 | 0.5 |
| Semantic_KMeans | 2015-02 | Topic 5 | 1 | 3 | 2 | 0.3333333333333333 | 0.5 |
| Semantic_KMeans | 2015-02 | UNASSIGNED | 1 | 3 | 2 | 0.3333333333333333 |  |
| Semantic_KMeans | 2015-03 | Topic 1 | 4 | 7 | 6 | 0.5714285714285714 | 0.6666666666666666 |
| Semantic_KMeans | 2015-03 | Topic 5 | 2 | 7 | 6 | 0.2857142857142857 | 0.3333333333333333 |
| Semantic_KMeans | 2015-03 | UNASSIGNED | 1 | 7 | 6 | 0.1428571428571428 |  |
| Semantic_KMeans | 2015-04 | Topic 1 | 8 | 16 | 15 | 0.5 | 0.5333333333333333 |
| Semantic_KMeans | 2015-04 | Topic 5 | 7 | 16 | 15 | 0.4375 | 0.4666666666666667 |
| Semantic_KMeans | 2015-04 | UNASSIGNED | 1 | 16 | 15 | 0.0625 |  |
| Semantic_KMeans | 2015-10 | Topic 1 | 5 | 40 | 38 | 0.125 | 0.131578947368421 |
| Semantic_KMeans | 2015-10 | Topic 2 | 1 | 40 | 38 | 0.025 | 0.0263157894736842 |
| Semantic_KMeans | 2015-10 | Topic 4 | 11 | 40 | 38 | 0.275 | 0.2894736842105263 |
| Semantic_KMeans | 2015-10 | Topic 5 | 21 | 40 | 38 | 0.525 | 0.5526315789473685 |
| Semantic_KMeans | 2015-10 | UNASSIGNED | 2 | 40 | 38 | 0.05 |  |
| Semantic_KMeans | 2016-01 | Topic 2 | 1 | 37 | 37 | 0.027027027027027 | 0.027027027027027 |
| Semantic_KMeans | 2016-01 | Topic 4 | 18 | 37 | 37 | 0.4864864864864865 | 0.4864864864864865 |
| Semantic_KMeans | 2016-01 | Topic 5 | 18 | 37 | 37 | 0.4864864864864865 | 0.4864864864864865 |
| Semantic_KMeans | 2016-02 | Topic 1 | 1125 | 3899 | 3752 | 0.2885355219286997 | 0.2998400852878465 |
| Semantic_KMeans | 2016-02 | Topic 2 | 674 | 3899 | 3752 | 0.1728648371377276 | 0.179637526652452 |
| Semantic_KMeans | 2016-02 | Topic 3 | 687 | 3899 | 3752 | 0.1761990253911259 | 0.1831023454157782 |
| Semantic_KMeans | 2016-02 | Topic 4 | 472 | 3899 | 3752 | 0.1210566812003077 | 0.1257995735607676 |
| Semantic_KMeans | 2016-02 | Topic 5 | 794 | 3899 | 3752 | 0.2036419594767889 | 0.2116204690831556 |
| Semantic_KMeans | 2016-02 | UNASSIGNED | 147 | 3899 | 3752 | 0.03770197486535 |  |


## Topic dictionary status

Topic-group records: 40.
Recorded decisions: accept: 35; do_not_label: 5

Groups marked `do_not_label` must not be reported as accepted topic labels.
Batch approval does not constitute independent example-level human validation.

## Limitations

- Topic labels are interpretations of model-generated clusters, not ground truth.
- VADER and transformer agreement measures agreement, not which method is necessarily correct.
- Statistical significance should be interpreted alongside effect sizes, sample sizes, assumptions, and multiple-testing corrections.
- The temporal dataset covers October 2014 to February 2016; observed periods are limited.
- Structural consistency does not independently establish statistical validity.
