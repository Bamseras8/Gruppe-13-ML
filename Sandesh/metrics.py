# ==========================================================================================
# AI-GENERATED CODE: made with an AI assistant (Claude Code).
# Reference copy only. If any of it is used, it must be stated in the report's AI declaration.
# ==========================================================================================
"""Detection and counting metrics.

All experiments are scored by these same functions, so the comparison is fair no matter
how the detections were produced (full image or sliced inference).

A "record" is one image: dict(gt_boxes, gt_cls, boxes, conf, cls) with boxes in pixel xyxy.

Detection : Precision, Recall, F1 (at a confidence threshold, IoU >= 0.5), AP / mAP@0.5 and mAP@0.5:0.95
Counting  : MAE and RMSE between predicted and true number of vehicles per image (total and per class)
"""
from __future__ import annotations

import numpy as np

from common import CLASS_NAMES, NC, SIZE_BINS

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between boxes a [M,4] and b [N,4] (xyxy)."""
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match(rec: dict, class_aware: bool = True, iou_thresholds=IOU_THRESHOLDS):
    """Greedy COCO-style matching: detections in order of confidence each take the best still-free
    ground-truth box with IoU >= threshold (and the same class if class_aware).

    Returns tp [N_det, T] (bool) and, for the first threshold, the matched GT index of each detection (-1 = none).
    """
    n_det, n_gt = len(rec["boxes"]), len(rec["gt_boxes"])
    tp = np.zeros((n_det, len(iou_thresholds)), dtype=bool)
    matched_gt = np.full(n_det, -1)
    if n_det == 0 or n_gt == 0:
        return tp, matched_gt
    iou = box_iou(rec["gt_boxes"], rec["boxes"])  # [M, N]
    if class_aware:
        iou = iou * (rec["gt_cls"][:, None] == rec["cls"][None, :])
    order = np.argsort(-rec["conf"], kind="stable")
    for t, thr in enumerate(iou_thresholds):
        free = np.ones(n_gt, dtype=bool)
        candidates = order[(iou[:, order] >= thr).any(axis=0)]
        for d in candidates:
            ious = np.where(free, iou[:, d], -1)
            g = int(ious.argmax())
            if ious[g] >= thr:
                free[g] = False
                tp[d, t] = True
                if t == 0:
                    matched_gt[d] = g
    return tp, matched_gt


def prepare(records: list[dict]):
    """Match every image once and cache the results inside the records."""
    for r in records:
        if "tp" not in r:
            r["tp"], r["match"] = match(r, class_aware=True)
            r["tp_agn"], r["match_agn"] = match(r, class_aware=False, iou_thresholds=IOU_THRESHOLDS[:1])
    return records


def _concat(records, key):
    return np.concatenate([r[key] for r in records]) if records else np.zeros(0)


# --------------------------------------------------------------------------- average precision
def average_precision(tp: np.ndarray, conf: np.ndarray, n_gt: int) -> np.ndarray:
    """AP for each IoU threshold (columns of tp), 101-point interpolated like COCO."""
    if n_gt == 0:
        return np.full(tp.shape[1], np.nan)
    if len(conf) == 0:
        return np.zeros(tp.shape[1])
    order = np.argsort(-conf, kind="stable")
    tp = tp[order]
    tpc, fpc = np.cumsum(tp, axis=0), np.cumsum(~tp, axis=0)
    recall, precision = tpc / n_gt, tpc / (tpc + fpc)
    ap = np.zeros(tp.shape[1])
    grid = np.linspace(0, 1, 101)
    for t in range(tp.shape[1]):
        # precision envelope (monotonically decreasing), then sample at 101 recall points
        p = np.maximum.accumulate(precision[::-1, t])[::-1]
        idx = np.searchsorted(recall[:, t], grid, side="left")
        ap[t] = np.mean(np.where(idx < len(p), p[np.minimum(idx, len(p) - 1)], 0))
    return ap


def ap_per_class(records):
    """AP [NC, 10] (IoU 0.5 ... 0.95) per class."""
    tp, conf, cls = _concat(records, "tp").reshape(-1, len(IOU_THRESHOLDS)), _concat(records, "conf"), \
        _concat(records, "cls").astype(int)
    gt_cls = _concat(records, "gt_cls").astype(int)
    return np.array([average_precision(tp[cls == c], conf[cls == c], int((gt_cls == c).sum())) for c in range(NC)])


def agnostic_ap50(records):
    """AP@0.5 when the vehicle type is ignored - how well vehicles are *found*, regardless of class."""
    tp = _concat(records, "tp_agn").reshape(-1, 1)
    return float(average_precision(tp, _concat(records, "conf"), len(_concat(records, "gt_cls")))[0])


# --------------------------------------------------------------------------- P / R / F1 at a threshold
def prf_at(records, conf_thr: float):
    """Per-class TP, FP, FN, precision, recall, F1 at IoU 0.5 for detections with conf >= conf_thr.
    (Matching is done in confidence order, so dropping low-confidence detections does not change
    the matches of the remaining ones.)"""
    tp50, conf = _concat(records, "tp").reshape(-1, len(IOU_THRESHOLDS))[:, 0], _concat(records, "conf")
    cls, gt_cls = _concat(records, "cls").astype(int), _concat(records, "gt_cls").astype(int)
    keep = conf >= conf_thr
    tp = np.bincount(cls[keep & tp50], minlength=NC)
    fp = np.bincount(cls[keep & ~tp50], minlength=NC)
    fn = np.bincount(gt_cls, minlength=NC) - tp
    p = tp / np.maximum(tp + fp, 1)
    r = tp / np.maximum(tp + fn, 1)
    f1 = 2 * p * r / np.maximum(p + r, 1e-9)
    micro_p = tp.sum() / max(tp.sum() + fp.sum(), 1)
    micro_r = tp.sum() / max(tp.sum() + fn.sum(), 1)
    micro_f1 = 2 * micro_p * micro_r / max(micro_p + micro_r, 1e-9)
    return dict(tp=tp, fp=fp, fn=fn, p=p, r=r, f1=f1, micro_p=micro_p, micro_r=micro_r, micro_f1=micro_f1)


def best_threshold(records, grid=np.round(np.arange(0.05, 0.951, 0.01), 2)):
    """Confidence threshold that maximises the overall (micro) F1. Chosen on the validation set."""
    f1 = [prf_at(records, t)["micro_f1"] for t in grid]
    return float(grid[int(np.argmax(f1))]), np.array(f1), grid


# --------------------------------------------------------------------------- counting
def counts(records, conf_thr: float):
    """Per-image true and predicted vehicle counts per class: arrays [n_images, NC]."""
    true = np.array([np.bincount(r["gt_cls"].astype(int), minlength=NC) for r in records])
    pred = np.array([np.bincount(r["cls"][r["conf"] >= conf_thr].astype(int), minlength=NC) for r in records])
    return true, pred


def counting_metrics(records, conf_thr: float):
    true, pred = counts(records, conf_thr)
    err_total = pred.sum(1) - true.sum(1)
    err_cls = pred - true
    return dict(
        mae_total=float(np.abs(err_total).mean()), rmse_total=float(np.sqrt((err_total ** 2).mean())),
        bias_total=float(err_total.mean()),  # > 0: over-counting, < 0: under-counting
        exact_total_percent=float((err_total == 0).mean() * 100),
        mae_class=np.abs(err_cls).mean(0), rmse_class=np.sqrt((err_cls ** 2).mean(0)),
        mae_class_mean=float(np.abs(err_cls).mean()), rmse_class_mean=float(np.sqrt((err_cls ** 2).mean())),
    )


# --------------------------------------------------------------------------- analysis
def recall_by_size(records, conf_thr: float):
    """Class-agnostic recall (IoU >= 0.5) for small / medium / large vehicles."""
    found_all, sizes = [], []
    for r in records:
        keep = (r["conf"] >= conf_thr) & (r["match_agn"] >= 0)
        found = np.zeros(len(r["gt_boxes"]), dtype=bool)
        found[r["match_agn"][keep]] = True
        wh = r["gt_boxes"][:, 2:] - r["gt_boxes"][:, :2]
        sizes.append(np.sqrt(wh[:, 0] * wh[:, 1]))
        found_all.append(found)
    found_all, sizes = np.concatenate(found_all), np.concatenate(sizes)
    out = {}
    for name, (lo, hi) in SIZE_BINS.items():
        m = (sizes >= lo) & (sizes < hi)
        out[name] = dict(recall=float(found_all[m].mean()) if m.any() else float("nan"), n=int(m.sum()))
    return out


def confusion_matrix(records, conf_thr: float):
    """Rows: true class (+ last row 'background' = false detections).
    Columns: predicted class (+ last column 'missed' = vehicle not detected).
    Detections are matched to ground truth ignoring class (IoU >= 0.5), so the matrix shows which
    types get confused with each other."""
    cm = np.zeros((NC + 1, NC + 1), dtype=int)
    for r in records:
        keep = r["conf"] >= conf_thr
        matched = np.zeros(len(r["gt_boxes"]), dtype=bool)
        for d in np.nonzero(keep)[0]:
            g = r["match_agn"][d]
            if g >= 0:
                cm[r["gt_cls"][g], r["cls"][d]] += 1
                matched[g] = True
            else:
                cm[NC, r["cls"][d]] += 1
        for g in np.nonzero(~matched)[0]:
            cm[r["gt_cls"][g], NC] += 1
    return cm


def summarise(val_records, test_records):
    """All metrics for one experiment. The confidence threshold is chosen on validation data only."""
    prepare(val_records)
    prepare(test_records)
    thr, _, _ = best_threshold(val_records)
    ap = ap_per_class(test_records)
    prf = prf_at(test_records, thr)
    cnt = counting_metrics(test_records, thr)
    summary = dict(
        conf_threshold=thr,
        precision=float(prf["micro_p"]), recall=float(prf["micro_r"]), f1=float(prf["micro_f1"]),
        map50=float(np.nanmean(ap[:, 0])), map50_95=float(np.nanmean(ap.mean(1))),
        agnostic_ap50=agnostic_ap50(test_records),
        count_mae=cnt["mae_total"], count_rmse=cnt["rmse_total"], count_bias=cnt["bias_total"],
        count_exact_percent=cnt["exact_total_percent"],
        count_mae_per_class=cnt["mae_class_mean"], count_rmse_per_class=cnt["rmse_class_mean"],
        recall_by_size=recall_by_size(test_records, thr),
    )
    per_class = [dict(cls=CLASS_NAMES[c], n_test=int(prf["tp"][c] + prf["fn"][c]), precision=float(prf["p"][c]),
                      recall=float(prf["r"][c]), f1=float(prf["f1"][c]), ap50=float(ap[c, 0]),
                      ap50_95=float(ap[c].mean()), count_mae=float(cnt["mae_class"][c]),
                      count_rmse=float(cnt["rmse_class"][c])) for c in range(NC)]
    return summary, per_class
