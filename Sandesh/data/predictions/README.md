# Predictions for the round 2 runs (v3 data)

The test-set predictions of our round 2 YOLO runs: every predicted box with its vehicle class and confidence score. One CSV file per run.

From Daniel, 27 October 2026. The models were trained and the predictions made on 27 October (RTX 4090 laptop).

## The three runs

All three are **YOLO11s** (Ultralytics), trained for 80 epochs on **Julius's train/val/test split and cleaned labels**.
They start from the COCO-pretrained `yolo11s.pt`. There is no `tiled` run in round 2.

| File | Training input | Inference on the test images | Boxes in file |
|---|---|---|---|
| `baseline_test_predictions.csv` | full image downscaled 1024 → 640 px | full image, 640 px | 6,792 |
| `hires_test_predictions.csv` | full image at native 1024 px | full image, 1024 px | 6,937 |
| `multiscale_test_predictions.csv` | 640 px ± 50 % (random 320–960 px per batch) | full image, 640 px | 7,527 |

Every file covers all **706 images of Julius's test split**. Compare them with Julius's **cleaned** test labels
(the ones the models were trained on), not with the original TRANSSET labels.

`hires` is the main model. The other runs were set up as small-vehicle experiments, which the teacher has removed from the scope. Which runs go in the report is a group decision, so all three are included.

**Do not mix these files with the round 1 predictions (`Predictions_five_runs`).** Round 1 used another split, so the test images are different.

## File format

One row per predicted box:

```
image,x1,y1,x2,y2,confidence,class_id,class_name
file_00008,390.38007,197.49477,551.13885,352.04895,0.90408736,6,truck
```

| Column | Meaning |
|---|---|
| `image` | Image name without `.jpg`, the same as in the test split list |
| `x1, y1, x2, y2` | Box corners in **pixels on the original 1024 × 1024 image**; (0, 0) is the top-left corner |
| `confidence` | Model score, 0–1 |
| `class_id`, `class_name` | Predicted vehicle class |

## Things that are easy to get wrong

1. **The files hold every box with confidence ≥ 0.001**, not only the "real" detections. Keeping the low scores is needed for mAP. For precision, recall, F1 and counting, **first drop the boxes below a confidence threshold**. Otherwise every run counts roughly twice as many vehicles as there are. Choose the threshold on the validation set, not on the test set.
2. **Class order.** The class table in the dataset README is wrong. The real order (used in the label files and here) is:
   `0 car, 1 hatchback, 2 pickup truck, 3 sedan, 4 suv, 5 trailer, 6 truck, 7 van`
3. **NMS is class-agnostic.** One vehicle gets at most one box, never two boxes with two different classes. Other settings: IoU 0.7, at most 300 boxes per image.
4. **Do not round the boxes.** The numbers are stored at full precision, and rounding can move a box across a size limit (for example small/medium).

## Not in this folder

- **The trained models** (`best.pt`, about 19 MB each). They are in `Trained_models_v3.zip`, with a README on how to run them.
