from pathlib import Path
import json
import sys

import torch
import ultralytics
import yaml
from ultralytics import RTDETR


DATA = Path(r"C:\ai\Code\datasets\transset_v3\data.yaml")

PROJECT = Path(__file__).resolve().parent / "runs_rtdetr"

#First run: True. For full training after a successful test, change this to: False.
SMOKE_TEST = False

#A pretty low batch size to start with
#Change to 1 if CUDA runs out of memory
BATCH_SIZE = 2


def main():
    if not DATA.is_file():
        raise FileNotFoundError(f"Cannot find dataset config: {DATA}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable. Check the Python environment.")

    expected_classes = [
        "car", "hatchback", "pickup truck", "sedan",
        "suv", "trailer", "truck", "van"
    ]

    yaml_text = DATA.read_text(encoding="utf-8-sig")
    dataset = yaml.safe_load(yaml_text)
    names = dataset["names"]
    if isinstance(names, dict):
        names = [names[i] for i in range(8)]
    if names != expected_classes:
        raise ValueError("Dataset class order differs from Daniels order")

    print("Python:", sys.executable)
    print("PyTorch:", torch.__version__)
    print("Ultralytics:", ultralytics.__version__)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Dataset:", DATA)

    if SMOKE_TEST:
        epochs = 1
        fraction = 0.02
        run_name = "rtdetr_1024_smoke"
        close_mosaic = 0
        print("SMOKE TEST: 1 epoch, 2% of training images, full validation set.")
    else:
        epochs = 80
        fraction = 1.0
        run_name = "rtdetr_1024"
        close_mosaic = 10
        print("FULL TRAINING: up to 80 epochs, all training images.")

    model = RTDETR("rtdetr-l.pt")

    model.train(
        data=str(DATA),
        imgsz=1024,
        epochs=epochs,
        patience=15,
        batch=BATCH_SIZE,
        device=0,
        workers=4,
        cache=False,
        fraction=fraction,
        seed=0,
        deterministic=False,

        # Initial optimizer settings chosen for RT-DETR fine-tuning.
        optimizer="AdamW",
        lr0=0.0001,
        lrf=0.01,
        weight_decay=0.0001,
        warmup_epochs=3.0,
        cos_lr=False,

        #Match the main geometric/color augmentation settings in YOLO hires.
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        mosaic=1.0,
        close_mosaic=close_mosaic,
        mixup=0.0,
        copy_paste=0.0,
        multi_scale=False,

        #FP32 avoids the documented RT-DETR AMP instability.
        amp=False,
        val=True,
        split="val",
        plots=True,
        save=True,
        project=str(PROJECT),
        name=run_name,
        exist_ok=False,
    )

    #Saves args.yaml, results.csv, best.pt and last.pt.
    run_folder = Path(model.trainer.save_dir)
    environment = {
        "python": sys.version,
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "dataset_yaml": str(DATA),
        "smoke_test": SMOKE_TEST,
        "note": "Validation during training; test-set evaluation is separate.",
    }
    (run_folder / "environment.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )

    print("\nFinished. Results:", run_folder)
    print("Best model:", run_folder / "weights" / "best.pt")
    if SMOKE_TEST:
        print("Set SMOKE_TEST = False and run again.")

if __name__ == "__main__":
    main()
