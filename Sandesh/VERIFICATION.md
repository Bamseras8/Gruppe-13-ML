# Checks performed on this rebuilt package

- Daniel, Julius and original text-label archives passed ZIP integrity checks.
- All 4,704 original text-label files were checked: 21,661 original boxes.
- Julius's train/validation/test split was recreated from his supplied grouping CSV and seed: 3,292 / 706 / 706. No image ID or supplied t8 group spans splits. This does not independently prove that every visually similar scene was grouped correctly.
- All 745 images on Julius's rotated-image list belong to training. Image rotation detection itself was not rerun in this rebuild.
- Julius's cleaning function reproduced 5 zero-size removals, 24 thin-box removals and 9 edge-clipping cases. All 706 packaged test labels match its six-decimal output, containing 3,299 boxes.
- The prediction CSVs are exact copies of Daniel's v3 exports: baseline 6,792 rows; hires 6,937; multiscale 7,527. All match the new test-image list.
- Ten tests cover correct detections, duplicate predictions, wrong classes, empty inputs, threshold boundaries, IoU, AP and regression results.
- A separate plain-Python implementation agrees on detection matching at all ten IoU thresholds, per-class AP, TP/FP/FN and confusion matrices for all three models.
- Counting MAE and RMSE were also checked with separate ordinary-Python arithmetic.
- Three evaluation runs produced identical result CSVs. The rebuilt ZIP was extracted into a separate directory and its download checker, tests and evaluator were run there.

These checks reduce the chance of errors; they do not guarantee that no error is possible. They validate the saved-box evaluation, not model retraining, fresh full-dataset inference, the original annotation quality or independent selection of the v3 confidence thresholds.

The earlier ZIP failed Python's archive-open check. This version was rebuilt from the intact source folders, not repaired by guessing missing bytes.
