# LeakSense Twin — Model Benchmark Results

Dataset: `data/training_data.csv` (50005 samples, 31 features). Test set: 7501 samples.

`dual` = separate binary detector + separate 6-class zone localizer (today's pattern).
`single` = one 6-class classifier; binary leak flag and confidence derived from it.

```
                                  name pattern  accuracy  macro_f1  leak_precision  leak_recall  leak_f1  latency_mean_us  latency_p95_us       size     train_s
LeakSenseNet+LocalizationNet (current)    dual  0.961472  0.901545        0.900344     0.873333 0.886633       102.833333      125.625003    76.6 KB  120.564912
                          Standard MLP  single  0.960672  0.899525        0.994295     0.813333 0.894756       469.093667      640.765003    57.9 KB  724.557103
                              LightGBM    dual  0.946407  0.868600        0.912356     0.846667 0.878285      1602.732667     2224.725001  4780.6 KB    6.447006
                              LightGBM  single  0.946407  0.868600        0.897831     0.855333 0.876067      1698.833666     2326.019997  4096.2 KB    3.344453
                               XGBoost    dual  0.939741  0.857376        0.931649     0.836000 0.881237      1180.791334     1999.725003  4773.2 KB   12.168425
                               XGBoost  single  0.939741  0.857376        0.857702     0.876000 0.866755       982.011333     1400.019999  3987.4 KB    6.800969
            GradientBoosting (sklearn)    dual  0.936408  0.849087        0.905865     0.834000 0.868448       269.761333      488.039993  5256.5 KB 1016.257010
            GradientBoosting (sklearn)  single  0.936408  0.849087        0.852168     0.864667 0.858372      1246.279000     1961.369997  4436.8 KB  690.891319
                          RandomForest  single  0.921610  0.813545        0.848240     0.771333 0.807961     99747.616667   184950.335000 10441.7 KB   17.191468
                          RandomForest    dual  0.921610  0.813545        0.980465     0.702667 0.818641    116309.762667   214233.634995 16268.9 KB    7.738573
                   Logistic Regression  single  0.869084  0.765898        0.627346     0.936000 0.751204       108.714000      152.135001     2.4 KB   22.173700
                   Logistic Regression    dual  0.869084  0.765898        0.718011     0.789333 0.751985       224.926333      525.280003     3.5 KB   26.757174
```
