"""Check the reference matcher and AP using a separate plain-Python calculation."""
import math
import numpy as np
import metrics
from evaluate import load_records, THRESHOLDS


def iou(a, b):
    width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection + 1e-9)


def match(record, threshold, class_aware=True):
    truth = record['gt_boxes'].tolist()
    predicted = record['boxes'].tolist()
    used = set()
    matches = [-1] * len(predicted)
    order = sorted(range(len(predicted)), key=lambda i: -float(record['conf'][i]))
    for d in order:
        candidates = [g for g in range(len(truth)) if g not in used and
                      (not class_aware or record['gt_cls'][g] == record['cls'][d])]
        if not candidates:
            continue
        best = max(candidates, key=lambda g: iou(truth[g], predicted[d]))
        if iou(truth[best], predicted[d]) >= threshold:
            matches[d] = best
            used.add(best)
    return matches


def ap_101(detections, n_truth):
    if not n_truth:
        return float('nan')
    ordered = sorted(detections, key=lambda row: -row[0])
    tp = 0
    curve = []
    for rank, (_, correct) in enumerate(ordered, 1):
        tp += correct
        curve.append((tp / n_truth, tp / rank))
    return sum(max((p for r, p in curve if r >= n / 100), default=0)
               for n in range(101)) / 101


def main():
    for model, threshold in THRESHOLDS.items():
        records = metrics.prepare(load_records(model))
        detections = [[[] for _ in range(10)] for _ in range(8)]
        totals, tp, fp = [0]*8, [0]*8, [0]*8
        matrix = [[0]*9 for _ in range(9)]
        for record in records:
            for c in record['gt_cls']:
                totals[int(c)] += 1
            for t, limit in enumerate(metrics.IOU_THRESHOLDS):
                matches = match(record, float(limit))
                assert [g >= 0 for g in matches] == record['tp'][:, t].tolist()
                for d, g in enumerate(matches):
                    c = int(record['cls'][d])
                    conf = float(record['conf'][d])
                    detections[c][t].append((conf, g >= 0))
                    if t == 0 and conf >= threshold:
                        tp[c] += g >= 0
                        fp[c] += g < 0
            matches = match(record, .5, class_aware=False)
            found = set()
            for d, g in enumerate(matches):
                if float(record['conf'][d]) < threshold:
                    continue
                c = int(record['cls'][d])
                if g >= 0:
                    matrix[int(record['gt_cls'][g])][c] += 1
                    found.add(g)
                else:
                    matrix[8][c] += 1
            for g, c in enumerate(record['gt_cls']):
                if g not in found:
                    matrix[int(c)][8] += 1
        expected_ap = metrics.ap_per_class(records)
        for c in range(8):
            for t in range(10):
                assert math.isclose(ap_101(detections[c][t], totals[c]), expected_ap[c, t], abs_tol=1e-12)
        scores = metrics.prf_at(records, threshold)
        assert tp == scores['tp'].tolist() and fp == scores['fp'].tolist()
        assert [totals[c] - tp[c] for c in range(8)] == scores['fn'].tolist()
        assert np.array_equal(matrix, metrics.confusion_matrix(records, threshold))
        print(model + ': separate matching, AP, TP/FP/FN and confusion checks passed')


if __name__ == '__main__':
    main()
