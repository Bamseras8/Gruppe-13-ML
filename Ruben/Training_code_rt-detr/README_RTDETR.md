# RT-DETR – training on TRANSSET

This project fine-tunes a COCO-pretrained RT-DETR-L on the eight vehicle classes in TRANSSET. The goal is to compare it with Daniel's YOLO11s model, particularly the `hires` model at an image resolution of 1024 px.

The training script is `train_rtdetr.py`. It uses the existing Ultralytics implementation of RT-DETR. The library handles the architecture, loss calculation and model weight updates.

This document describes the planned training setup. It contains no final RT-DETR results, and the setup has not yet been verified by a training run on Daniel's GPU.

## Dataset

We use Daniel's existing, prepared `transset_v3`, including Julius's split lists and the label corrections described in Daniel's README. The dataset should not be prepared or split again before this training run.

| Split | Images |
| --- | ---: |
| Train | 3,292 |
| Val | 706 |
| Test | 706 |

The images are 1024 × 1024 px. The dataset version describes the data, rather than the model trained on it. The name `transset_v3` therefore also applies when training RT-DETR.

The class IDs must follow this order:

| ID | Class |
| --- | --- |
| 0 | car |
| 1 | hatchback |
| 2 | pickup truck |
| 3 | sedan |
| 4 | suv |
| 5 | trailer |
| 6 | truck |
| 7 | van |

The training script checks the class order in `data.yaml`. This check does not verify the complete dataset contents or the splits.

## Before running

Training will run on Daniel's PC using an NVIDIA GPU and a Python environment with CUDA-compatible PyTorch, Ultralytics and PyYAML. The script stops if PyTorch cannot find a CUDA GPU.

Daniel's YOLO models were trained with Ultralytics 8.4.171. Record the version actually used for RT-DETR. The script prints the versions in the terminal and saves them in `environment.json` after the training call finishes.

Check this path in `train_rtdetr.py`:

```python
DATA = Path(r"C:\ai\Code\datasets\transset_v3\data.yaml")
```

The paths inside `data.yaml` must also be correct for the PC. Internet access is required on the first run if `rtdetr-l.pt` needs to be downloaded.

## 1. Run a smoke test

Start with:

```python
SMOKE_TEST = True
BATCH_SIZE = 2
```

Run the script from PyCharm or a terminal using the correct Python environment:

```bash
python train_rtdetr.py
```

The smoke test is configured for one epoch with `fraction=0.02`, which limits training to approximately 2% of the training images in the intended library version. Check the image counts reported in the terminal; argument handling can change between library versions. The code is intended to validate on the complete validation set.

The purpose is to check that the dataset loads, the GPU works, training completes and the outputs are saved. Smoke-test scores must not be used as the model's final results in the report.

If a CUDA out-of-memory error occurs, reduce `BATCH_SIZE` to `1`. Record the batch size actually used.

## 2. Run full training

After a successful smoke test, change:

```python
SMOKE_TEST = False
```

Then run the same script again. It uses the complete training set and trains for up to 80 epochs. With `patience=15`, training can stop early after 15 epochs without improvement in the validation metric monitored by the library.

**Full training does not continue from the smoke-test model.** Both runs load the original COCO-pretrained weights from `rtdetr-l.pt`. Full training therefore starts from pretrained weights, rather than random weights or the smoke test's `best.pt`.

Running the script again also starts a new training run; the script does not include a function for resuming an interrupted run.

## Training settings and rationale

The settings combine Daniel's setup, common library settings and suggested starting values. They have not been optimized through a hyperparameter search on TRANSSET.

| Setting | Value | Rationale |
| --- | --- | --- |
| Model | `rtdetr-l.pt` | COCO-pretrained RT-DETR-L |
| Image resolution | `1024` | Same resolution as YOLO `hires` and the original images |
| Maximum epochs | `80` | Same upper limit as Daniel's training |
| Patience | `15` | Same setting as Daniel's training |
| Batch size | `2` | Conservative starting value for GPU memory use; not an optimized value |
| Optimizer | `AdamW` | Suggested starting point for fine-tuning |
| Initial learning rate | `0.0001` | Suggested starting value; must be assessed using validation results |
| Final learning rate factor | `0.01` | `lrf`; retained as a starting point |
| Weight decay | `0.0001` | Suggested regularization; not optimized |
| Warmup | `3.0` epochs | Gradual start to training |
| Cosine scheduler | `False` | Does not use a cosine learning rate schedule |
| Seed | `0` | Fixed seed, as in Daniel's setup |
| Deterministic operations | `False` | RT-DETR documentation describes a limitation involving `grid_sample` |
| AMP | `False` | Conservative choice because RT-DETR documentation describes possible NaN instability with AMP |
| Workers | `4` | Starting value for data loading |
| Cache | `False` | Avoids caching all images in RAM |

The augmentation arguments match Daniel's `hires` settings:

```python
degrees=0.0
translate=0.1
scale=0.5
shear=0.0
perspective=0.0
flipud=0.0
fliplr=0.5
hsv_h=0.015
hsv_s=0.7
hsv_v=0.4
mosaic=1.0
close_mosaic=10
mixup=0.0
copy_paste=0.0
multi_scale=False
```

The smoke test uses `close_mosaic=0`. Matching argument values does not necessarily mean that two model implementations perform identical image processing.

RT-DETR-L and YOLO11s differ in architecture and size. The setup matches the data, resolution and several training conditions, but does not provide an identical computational budget or identical optimizer settings. These differences must be documented in the report. An 80-epoch limit also does not guarantee that RT-DETR has stopped improving.

## Where are the outputs saved?

Outputs are saved in a separate directory beside `train_rtdetr.py`:

```text
runs_rtdetr/
    rtdetr_1024_smoke/
    rtdetr_1024/
        weights/
            best.pt
            last.pt
        args.yaml
        results.csv
        environment.json
        ...plots and other training outputs
```

| File | Purpose |
| --- | --- |
| `weights/best.pt` | Model selected according to the library's validation criterion; used for further evaluation |
| `weights/last.pt` | Last saved checkpoint |
| `args.yaml` | Training settings saved by the library |
| `results.csv` | Training and validation measurements for each epoch |
| `environment.json` | Python, PyTorch, Ultralytics and GPU information, written after the training call |

`exist_ok=False` is set to avoid reusing the same output directory. The library may assign a numbered directory name to subsequent runs. Use the actual path printed in the terminal.

## Comparison with YOLO

The script performs validation during training. It does not perform the final test-set evaluation or the comparison with YOLO.

After training, use both models' `best.pt` files on the same 706 test images with the same ground-truth annotations. Use Daniel's evaluation pipeline as a starting point and check that it handles RT-DETR predictions correctly.

- Calculate mAP, precision, recall, F1 and counting errors using the same definitions and matching rules.
- Select a confidence threshold on the validation set for each model. Do not automatically reuse YOLO's threshold for RT-DETR.
- Clarify how overlapping predictions are handled. Daniel's reported YOLO results use class-agnostic NMS; RT-DETR has a different prediction pipeline, and `agnostic_nms=True` must not be assumed to produce equivalent processing.
- Use the validation set for training decisions and the test set for final evaluation.
- Measure any speed differences on the same machine using the same measurement procedure.

After training, Daniel can send back `best.pt`, `args.yaml`, `results.csv` and `environment.json`. The model files can be used on another PC with a compatible Python environment. Analysis can also be performed using exported predictions without running the model again each time.

## Documentation and AI assistance

The training script was developed with assistance from ChatGPT. It uses the Ultralytics implementation and is not a custom implementation of the RT-DETR architecture. Describe individual contributions and AI use in the submission according to the course requirements, and update the description to reflect the work actually completed.

References:

- [Ultralytics RT-DETR](https://docs.ultralytics.com/models/rtdetr/)
- [Ultralytics training settings](https://docs.ultralytics.com/modes/train/)
- [RT-DETR trainer: notes on AMP and determinism](https://docs.ultralytics.com/reference/models/rtdetr/train/)
- Daniel's YOLO11 README, “Trained models, round 2 (v3 data)”.

Online documentation describes the current library version and may differ from the version used in the experiment. Keep `args.yaml`, version information and the actual training script together with the results.
