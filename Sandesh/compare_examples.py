import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
IMAGES = ROOT / 'images'
OUTPUT = ROOT / 'results' / 'error_examples'
NAMES = ['car', 'hatchback', 'pickup', 'sedan', 'suv', 'trailer', 'truck', 'van']
THRESHOLDS = {'baseline': 0.45, 'hires': 0.41, 'multiscale': 0.40}
EXAMPLES = ['file_04130', 'file_02335', 'file_02068']


def read_labels(name, width, height):
    boxes = []
    for line in (DATA / 'labels' / f'{name}.txt').read_text().splitlines():
        if not line.strip():
            continue
        cls, x, y, w, h = map(float, line.split())
        boxes.append(((x-w/2)*width, (y-h/2)*height,
                      (x+w/2)*width, (y+h/2)*height, NAMES[int(cls)]))
    return boxes


def draw_panel(ax, picture, boxes, title, color, crop=None):
    ax.imshow(picture)
    for x1, y1, x2, y2, label in boxes:
        ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1,
                               fill=False, edgecolor=color, linewidth=1.3))
        ax.text(x1, y1 + 2, label, fontsize=5.5, color='white',
                va='top', clip_on=True,
                bbox=dict(facecolor=color, edgecolor='none', pad=0.6, alpha=0.85))
    if crop:
        ax.set_xlim(crop[0], crop[2])
        ax.set_ylim(crop[3], crop[1])
    ax.set_title(title, fontsize=12)
    ax.axis('off')


def main():
    predictions = {}
    for model, threshold in THRESHOLDS.items():
        path = DATA / 'predictions' / f'{model}_test_predictions.csv'
        with path.open(newline='', encoding='utf-8-sig') as file:
            predictions[model] = [r for r in csv.DictReader(file)
                                  if float(r['confidence']) >= threshold]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name in EXAMPLES:
        path = IMAGES / f'{name}.jpg'
        if not path.is_file():
            raise FileNotFoundError(f'Place the original image here: {path}')
        with Image.open(path) as source:
            picture = source.convert('RGB')
        if picture.size != (1024, 1024):
            raise ValueError('Prediction coordinates require the original 1024 x 1024 images.')
        labels = read_labels(name, *picture.size)
        panels = [(labels, f'Supplied cleaned labels: {len(labels)}', '#178747')]
        for model, threshold in THRESHOLDS.items():
            boxes = []
            for r in predictions[model]:
                if r['image'] == name:
                    box = tuple(float(r[k]) for k in ('x1', 'y1', 'x2', 'y2'))
                    boxes.append((*box, f"{NAMES[int(r['class_id'])]} {float(r['confidence']):.2f}"))
            panels.append((boxes, f'{model}: {len(boxes)} predictions (confidence >= {threshold:.2f})', '#2166ac'))
        # Detail views keep the same boxes and thresholds; only the view is cropped.
        crops = {'file_04130': (430, 0, 1024, 320),
                 'file_02335': (580, 0, 1024, 120),
                 'file_02068': (250, 0, 950, 120)}
        for suffix, crop in [('comparison', None), ('detail', crops[name])]:
            fig, axes = plt.subplots(2, 2, figsize=(14, 14 if crop is None else 7))
            for ax, (boxes, title, color) in zip(axes.flat, panels):
                draw_panel(ax, picture, boxes, title, color, crop)
            fig.suptitle(f'{name} | ' + ('Full image' if crop is None else 'Top-area detail (counts refer to full image)'), fontsize=16)
            fig.text(0.5, 0.015, 'Green: supplied labels. Blue: saved predictions. Classes are shown as supplied; labels and thresholds are unchanged.',
                     ha='center', fontsize=9)
            fig.tight_layout(rect=(0, 0.035, 1, 0.965))
            destination = OUTPUT / f'{name}_{suffix}.png'
            fig.savefig(destination, dpi=160)
            plt.close(fig)
            print(f'Saved: {destination}')
        print(name, 'labels/predictions:', [len(p[0]) for p in panels])
    print('Done: six comparison images saved in results/error_examples.')


if __name__ == '__main__':
    main()
