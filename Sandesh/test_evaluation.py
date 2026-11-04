import unittest
import numpy as np
import metrics
from evaluate import load_records, THRESHOLDS


def record(boxes, classes, truth, truth_classes):
    return dict(boxes=np.array(boxes, dtype=float).reshape(-1, 4),
                cls=np.array(classes, dtype=int), conf=np.linspace(.9, .6, len(boxes)),
                gt_boxes=np.array(truth, dtype=float).reshape(-1, 4),
                gt_cls=np.array(truth_classes, dtype=int))


class Checks(unittest.TestCase):
    def test_perfect(self):
        r = record([[0, 0, 10, 10]], [0], [[0, 0, 10, 10]], [0])
        result = metrics.prepare([r])
        self.assertEqual(metrics.prf_at(result, .5)['micro_f1'], 1)
        self.assertEqual(metrics.ap_per_class(result)[0, 0], 1)

    def test_duplicate(self):
        r = record([[0, 0, 10, 10]] * 2, [0, 0], [[0, 0, 10, 10]], [0])
        scores = metrics.prf_at(metrics.prepare([r]), .5)
        self.assertEqual(scores['tp'][0], 1)
        self.assertEqual(scores['fp'][0], 1)

    def test_wrong_class(self):
        r = record([[0, 0, 10, 10]], [1], [[0, 0, 10, 10]], [0])
        result = metrics.prepare([r])
        self.assertEqual(metrics.prf_at(result, .5)['micro_f1'], 0)
        self.assertEqual(metrics.confusion_matrix(result, .5)[0, 1], 1)

    def test_no_predictions(self):
        r = record([], [], [[0, 0, 10, 10]], [0])
        result = metrics.prepare([r])
        self.assertEqual(metrics.counting_metrics(result, .5)['mae_total'], 1)
        self.assertEqual(metrics.ap_per_class(result)[0, 0], 0)

    def test_no_vehicles(self):
        r = record([[0, 0, 10, 10]], [0], [], [])
        result = metrics.prepare([r])
        self.assertEqual(metrics.confusion_matrix(result, .5)[8, 0], 1)

    def test_both_empty(self):
        result = metrics.prepare([record([], [], [], [])])
        self.assertEqual(metrics.counting_metrics(result, .5)['mae_total'], 0)
        self.assertEqual(metrics.prf_at(result, .5)['micro_f1'], 0)

    def test_threshold_boundary(self):
        r = record([[0, 0, 10, 10]], [0], [[0, 0, 10, 10]], [0])
        r['conf'] = np.array([0.5])
        result = metrics.prepare([r])
        self.assertEqual(metrics.prf_at(result, .5)['tp'][0], 1)
        self.assertEqual(metrics.prf_at(result, .5001)['fn'][0], 1)

    def test_iou(self):
        a = np.array([[0, 0, 10, 10]], dtype=float)
        b = np.array([[5, 0, 15, 10], [20, 20, 30, 30]], dtype=float)
        result = metrics.box_iou(a, b)[0]
        self.assertAlmostEqual(result[0], 1 / 3)
        self.assertEqual(result[1], 0)

    def test_ap_false_positive_first(self):
        tp = np.array([[False], [True]])
        self.assertAlmostEqual(metrics.average_precision(tp, np.array([.9, .8]), 1)[0], .5)

    def test_real_data(self):
        expected = {'baseline': (.45042492917847027, .6447449293177628),
                    'hires': (.40509915014164305, .6622900763358779),
                    'multiscale': (.4688385269121813, .6353446952249348)}
        for model, threshold in THRESHOLDS.items():
            result = metrics.prepare(load_records(model))
            self.assertAlmostEqual(metrics.counting_metrics(result, threshold)['mae_total'], expected[model][0])
            self.assertAlmostEqual(metrics.prf_at(result, threshold)['micro_f1'], expected[model][1])
            confusion = metrics.confusion_matrix(result, threshold)
            true, pred = metrics.counts(result, threshold)
            self.assertEqual(confusion[:8, :].sum(), true.sum())
            self.assertEqual(confusion[:, :8].sum(), pred.sum())


if __name__ == '__main__':
    unittest.main()
