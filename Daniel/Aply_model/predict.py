from pathlib import Path
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
OUTPUT.mkdir(exist_ok=True)

model = YOLO(str(HERE / "model_Name.pt")) # <- change if other model
for path in sorted((HERE / "input").iterdir()):
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}: # <-- can add / remove it need different file types
        ting = model.predict(str(path), imgsz=1024, conf=0.30, agnostic_nms=True, verbose=False)[0] # <-- change conf to what is relevant for your model
        ting.save(filename=str(OUTPUT / path.name))
