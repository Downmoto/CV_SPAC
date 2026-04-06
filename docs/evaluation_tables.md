# evaluation tables

## detector metrics

| snapshot | epoch | precision | recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| best (by mAP50-95) | 1 | 0.0036 | 0.9592 | 0.4633 | 0.2003 |
| last recorded | 1 | 0.0036 | 0.9592 | 0.4633 | 0.2003 |

epochs recorded: 1

## pipeline base metrics

| metric | value |
|---|---:|
| samples | 44 |
| detections | 43 |
| detection_rate | 0.977273 |
| ocr_nonempty | 43 |
| ocr_nonempty_rate | 0.977273 |
| pred_access_granted | 10 |
| pred_access_denied | 34 |
| mean_detector_conf | 0.761268 |
| mean_ocr_conf | 0.592573 |

## decision metrics (labeled)

| metric | value |
|---|---:|
| labeled_samples | 44 |
| unlabeled_samples | 0 |
| positive_label | Access Granted |
| accuracy | 0.886364 |
| precision | 0.600000 |
| recall | 0.857143 |
| f1 | 0.705882 |

| confusion term | count |
|---|---:|
| tp | 6 |
| tn | 33 |
| fp | 4 |
| fn | 1 |

| plate metric | value |
|---|---:|
| plate_labeled_samples | 39 |
| plate_exact_match_rate | 0.307692 |

## artifact paths

- inference json: outputs\predictions\inference_results.json
- ground truth csv: outputs\metrics\ground_truth_template.csv
