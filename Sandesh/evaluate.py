"""Evaluate round 2 saved boxes using Daniel's reference metric functions."""
from pathlib import Path
import csv
import math
import numpy as np
import metrics
from common import CLASS_NAMES, read_yolo_labels, yolo_to_xyxy

ROOT = Path(__file__).resolve().parent
THRESHOLDS = {'baseline': 0.45, 'hires': 0.41, 'multiscale': 0.40}


def save_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_records(model):
    stems = (ROOT / 'data/test_images.txt').read_text().split()
    assert len(stems) == len(set(stems)) == 706
    predictions = {stem: [] for stem in stems}
    with (ROOT / f'data/predictions/{model}_test_predictions.csv').open() as file:
        for row in csv.DictReader(file):
            assert row['image'] in predictions
            cls = int(row['class_id'])
            assert 0 <= cls < 8 and CLASS_NAMES[cls] == row['class_name']
            values = [float(row[key]) for key in ['x1', 'y1', 'x2', 'y2', 'confidence']]
            assert all(math.isfinite(value) for value in values)
            x1, y1, x2, y2, conf = values
            assert 0 <= conf <= 1 and x1 < x2 and y1 < y2
            predictions[row['image']].append(values + [cls])
    records = []
    for stem in stems:
        labels = read_yolo_labels(ROOT / f'data/labels/{stem}.txt')
        if not np.isfinite(labels).all():
            raise ValueError(f'{stem}: non-finite label')
        if len(labels):
            if not ((labels[:, 0] >= 0) & (labels[:, 0] < 8) &
                    (labels[:, 0] == labels[:, 0].astype(int))).all():
                raise ValueError(f'{stem}: invalid class ID')
            if not ((labels[:, 1:] >= 0) & (labels[:, 1:] <= 1)).all():
                raise ValueError(f'{stem}: label outside normalized range')
            if not (labels[:, 3:] > 0).all():
                raise ValueError(f'{stem}: zero-size box')
        pred = np.array(predictions[stem], dtype=np.float32).reshape(-1, 6)
        records.append(dict(stem=stem, boxes=pred[:, :4], conf=pred[:, 4],
                            cls=pred[:, 5].astype(int), gt_boxes=yolo_to_xyxy(labels),
                            gt_cls=labels[:, 0].astype(int)))
    assert sum(len(r['gt_cls']) for r in records) == 3299
    return records


def main():
    output = ROOT / 'results'
    output.mkdir(exist_ok=True)
    summaries, images, categories, largest = [], [], [], []
    for model, threshold in THRESHOLDS.items():
        records = metrics.prepare(load_records(model))
        scores = metrics.prf_at(records, threshold)
        ap = metrics.ap_per_class(records)
        counts = metrics.counting_metrics(records, threshold)
        true, pred = metrics.counts(records, threshold)
        errors = pred.sum(1) - true.sum(1)
        # Check counting again using ordinary Python arithmetic.
        assert math.isclose(sum(abs(int(e)) for e in errors) / len(errors), counts['mae_total'])
        assert math.isclose(math.sqrt(sum(int(e)**2 for e in errors) / len(errors)), counts['rmse_total'])
        summaries.append(dict(model=model, images=len(records), threshold=threshold,
            precision=scores['micro_p'], recall=scores['micro_r'], f1=scores['micro_f1'],
            map50=float(np.nanmean(ap[:, 0])), map50_95=float(np.nanmean(ap)),
            mae=counts['mae_total'], rmse=counts['rmse_total'], bias=counts['bias_total'],
            under=int(sum(errors < 0)), exact=int(sum(errors == 0)), over=int(sum(errors > 0)),
            exact_percent=counts['exact_total_percent']))
        model_rows = []
        for i, record in enumerate(records):
            row = dict(model=model, image=record['stem'], actual=int(true[i].sum()),
                       predicted=int(pred[i].sum()), error=int(errors[i]), absolute_error=abs(int(errors[i])))
            for c, name in enumerate(CLASS_NAMES):
                row['true_' + name] = int(true[i, c])
                row['pred_' + name] = int(pred[i, c])
            model_rows.append(row)
        images.extend(model_rows)
        largest.extend(sorted(model_rows, key=lambda r: -r['absolute_error'])[:10])
        for c, name in enumerate(CLASS_NAMES):
            categories.append(dict(model=model, category=name, tp=int(scores['tp'][c]),
                fp=int(scores['fp'][c]), fn=int(scores['fn'][c]), precision=scores['p'][c],
                recall=scores['r'][c], f1=scores['f1'][c], ap50=ap[c, 0],
                map50_95=float(ap[c].mean()), mae=counts['mae_class'][c], rmse=counts['rmse_class'][c]))
        matrix = metrics.confusion_matrix(records, threshold)
        matrix_rows = [dict(actual=name, **dict(zip(CLASS_NAMES + ['missed'], matrix[i])))
                       for i, name in enumerate(CLASS_NAMES + ['background'])]
        save_csv(output / f'{model}_confusion_matrix.csv', matrix_rows)
        print(f'{model:12} Images: {len(records)}  MAE: {counts["mae_total"]:.3f}  '
              f'RMSE: {counts["rmse_total"]:.3f}  F1: {scores["micro_f1"]:.3f}')
    for name, rows in [('summary', summaries), ('image_counts', images),
                       ('category_results', categories), ('largest_count_errors', largest)]:
        save_csv(output / (name + '.csv'), rows)


if __name__ == '__main__':
    main()
