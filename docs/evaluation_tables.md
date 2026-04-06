# evaluation tables

## detector metrics

| snapshot | epoch | precision | recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| best (by mAP50-95) | 57 | 0.9722 | 0.8367 | 0.8753 | 0.5147 |
| last recorded | 80 | 0.9755 | 0.8110 | 0.8702 | 0.4634 |

epochs recorded: 80

## pipeline base metrics

| metric | value |
|---|---:|
| samples | 44 |
| detections | 43 |
| detection_rate | 0.977273 |
| ocr_nonempty | 43 |
| ocr_nonempty_rate | 0.977273 |
| pred_access_granted | 0 |
| pred_access_denied | 44 |
| mean_detector_conf | 0.761242 |
| mean_ocr_conf | 0.690077 |

## decision metrics (labeled)

| metric | value |
|---|---:|
| labeled_samples | 44 |
| unlabeled_samples | 0 |
| positive_label | Access Granted |
| accuracy | 1.000000 |
| precision | 0.000000 |
| recall | 0.000000 |
| f1 | 0.000000 |

| confusion term | count |
|---|---:|
| tp | 0 |
| tn | 44 |
| fp | 0 |
| fn | 0 |

| plate metric | value |
|---|---:|
| plate_labeled_samples | 39 |
| plate_exact_match_rate | 0.256410 |

## artifact paths

- inference json: outputs/predictions/inference_results.json
- ground truth csv: outputs/metrics/ground_truth_template.csv
