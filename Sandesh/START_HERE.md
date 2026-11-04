# Round 2 vehicle evaluation

Extract this entire folder beside your old vehicle_evaluation folder. Do not mix their data.

1. Open the folder in PyCharm.
2. Install the required packages: `python -m pip install -r requirements.txt`.
3. Run `verify_download.py`, then `test_evaluation.py`. All ten tests should pass.
4. Run `evaluate.py`.
5. Open `results/summary.csv` and `results/largest_count_errors.csv`.
6. Run `plot_results.py` to create the results graphs.
7. Run `compare_examples.py` to create comparison images using the three example images in the images folder.

Optional: run `cross_check.py`. This checks matching, AP and confusion matrices using a separate plain-Python implementation. It takes longer than the main evaluator.

Use Windows **Extract All** on the ZIP before opening the folder in PyCharm. Do not run files directly from inside the ZIP. The folder contains the code, three prediction CSVs, 706 label files and seven result CSVs. Images are only needed for the comparison figures. Trained weights are not required

This evaluates all three saved prediction CSVs against 706 cleaned test-label files containing 3,299 vehicles. Images and model weights are not needed for this calculation.

Expected MAE / RMSE / F1:

- baseline: 0.450 / 0.833 / 0.645
- hires: 0.405 / 0.760 / 0.662
- multiscale: 0.469 / 0.841 / 0.635

## Sources and limits

The input predictions are Daniel's round 2 (v3) exports. Labels were reconstructed by executing Julius's exact `rett_labels` function on the original labels, then writing six decimal places as his script does. The test-image list is Julius's supplied split. The original cleaning script is included for provenance; do not run it inside this package because its dataset paths are machine-specific.

`metrics.py` and `common.py` are copied from Daniel's supplied AI-generated reference evaluator, with attribution retained. `evaluate.py` and the tests were prepared with ChatGPT assistance. These are recalculated results using the reference metric implementation, not a claim of an independently authored detection evaluator. Follow your course's AI declaration requirements.

Confidence thresholds (baseline 0.45, hires 0.41, multiscale 0.40) come from Daniel's new-model README. New validation predictions were not supplied, so threshold selection has not been independently reproduced. Thresholds are not optimized on the test set.

Precision/recall/F1 use same-class matching at IoU >= 0.5. mAP uses the exported low-confidence predictions and 101-point interpolated AP across classes and IoU thresholds 0.50 to 0.95. This is the supplied custom evaluator, not full pycocotools COCO evaluation. Confusion matrices match without class restrictions: rows are actual classes plus background; columns are predicted classes plus missed.

The old 705-image results belong to a different split and must not be presented as these new results. These test scores do not establish performance on night photos, side-view photos or rotated inputs.

This package does not retrain models or rerun inference on all images. It verifies the saved predictions.
