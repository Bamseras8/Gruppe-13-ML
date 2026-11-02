"""
Train a YOLO11 vehicle detector for one experiment.

Setups:
baseline full image downscaled to 640 px (reference)
hires full image at native 1024 px (higher resolution)
multiscale 640 px, input size randomly varied 320-960 px (multiscale traning)

WHAT TO RUN IN TERMINAL:
python train.py --setup baseline --smoke-test      # small trial run to check that everything works
python train.py --setup x (baseline, hires, multiscale) --workers x (dependent on system, 8 is default) --epochs x (default is 80) --cache (ram, disk) (default=false

Needs the dataset (Form julius) datasets/transset_v3 and a GPU.
Weights are saved to runs_v3/<exp>/weights/best.pt.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

CODE_GO = Path(__file__).resolve().parent
MODEL_GO = CODE_GO / "weights" / "yolo11s.pt" # pretrained COCO weights, downloaded on first use. You can change the yolo11s.pt to what model you want for eks. yolo26s.pt.

DATA_GO = CODE_GO / "datasets" / "transset_v3" / "data.yaml"# full 1024 px images, Julius's split + label fixes
RUNS_HERE = CODE_GO / "runs_v3" # training runs: weights, per-epoch results, plots

SETUP = {
    "baseline": dict(imgsz=640, batch=16),
    "hires": dict(imgsz=1024, batch=8),
    "multiscale": dict(imgsz=640, batch=16, multi_scale=0.5),
}


def main():
    argu = argparse.ArgumentParser()
    argu.add_argument("--setup", required=True, choices=SETUP)
    argu.add_argument("--epochs", type=int, default=80)
    argu.add_argument("--workers", type=int, default=8)
    argu.add_argument("--cache", choices=("ram", "disk"), default=False)
    argu.add_argument("--smoke-test", action="store_true")
    args = argu.parse_args()

    name = f"{args.setup}_smoke" if args.smoke_test else args.setup
    train_args = dict(
        data=str(DATA_GO), epochs=args.epochs, patience=15, workers=args.workers, cache=args.cache,
        project=str(RUNS_HERE), name=name, exist_ok=True, seed=0, **SETUP[args.setup]
    )
    if args.smoke_test:
        train_args.update(epochs=1, imgsz=320, batch=8, fraction=0.03, workers=2, plots=False)

    MODEL_GO.parent.mkdir(exist_ok=True)
    YOLO(str(MODEL_GO)).train(**train_args)
    print(f"Done. Best weights: {RUNS_HERE / name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()

"""
When i did traing V3 i owerwrote the epochs 50 to 80 because on traing V1/V2
it was til improving at 50.
Training was done on a 4090 laptop.
"""
