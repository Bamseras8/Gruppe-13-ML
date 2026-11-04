from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- paths
CODE_DIR = Path(__file__).resolve().parent
_DEFAULT_RAW = CODE_DIR.parent / "INFO" / "DataSet_&_project_or_task_ditales" / "DataSet" / "TRANSSET_v2"

RAW_DIR = Path(os.environ.get("TRANSSET_ROOT", _DEFAULT_RAW))
WORK_DIR = Path(os.environ.get("TRANSSET_WORK", CODE_DIR / "datasets"))
RUNS_DIR = Path(os.environ.get("TRANSSET_RUNS", CODE_DIR / "runs"))
OUT_DIR = Path(os.environ.get("TRANSSET_OUTPUTS", CODE_DIR / "outputs"))
SPLITS_DIR = CODE_DIR / "splits"
WEIGHTS_DIR = CODE_DIR / "weights"

# --------------------------------------------------------------------------- dataset
# Class order used by the label files (label_txt/classes.txt, VOC and YOLO labels).
# NB: the table in the dataset README lists a different order (sedan=0, suv=1, ...),
# but the actual annotation files use this alphabetical order.
CLASS_NAMES = ["car", "hatchback", "pickup truck", "sedan", "suv", "trailer", "truck", "van"]
NC = len(CLASS_NAMES)
IMG_SIZE = 1024  # every TRANSSET image is 1024 x 1024 px
SPLITS = ("train", "val", "test")

FULL_DATASET = "transset_full"    # full 1024 px images
TILE_DATASET = "transset_tiles"   # 640 px tiles cut from the 1024 px images
TILE_SIZE = 640
TILE_OVERLAP = 0.25               # 25 % overlap -> 2 x 2 tiles per image (offsets 0 and 384 px)

# Object size bins (sqrt of box area in original 1024 px image), used to analyse small-object recall.
# COCO's "small" (< 32 px) holds only ~2 % of TRANSSET boxes, so dataset-specific bins are used.
SIZE_BINS = {"small (<48 px)": (0, 48), "medium (48-96 px)": (48, 96), "large (>=96 px)": (96, 1e9)}

# --------------------------------------------------------------------------- experiments
# One entry per training configuration. "baseline" is the reference; the others each
# change one thing to improve detection of small vehicles.
EXPERIMENTS = {
    "baseline": dict(
        description="Full image downscaled 1024->640 px (standard YOLO setting)",
        dataset=FULL_DATASET, imgsz=640, batch=16, epochs=50, train_args={},
        infer="full", infer_imgsz=640,
    ),
    "hires": dict(
        description="Higher-resolution input: full image at native 1024 px",
        dataset=FULL_DATASET, imgsz=1024, batch=8, epochs=50, train_args={},
        infer="full", infer_imgsz=1024,
    ),
    "multiscale": dict(
        description="Multi-scale training: input size varies 320-960 px (640 +/- 50 %)",
        dataset=FULL_DATASET, imgsz=640, batch=16, epochs=50, train_args={"multi_scale": 0.5},
        infer="full", infer_imgsz=640,
    ),
    "tiled": dict(
        description="Image tiling: trained on 640 px tiles at native resolution, sliced (SAHI) inference",
        # each image gives 4 tiles (4 x 640^2 px vs 1024^2 px): 25 tile epochs ~ 40 hires epochs of compute
        dataset=TILE_DATASET, imgsz=640, batch=16, epochs=25, train_args={},
        infer="sliced", infer_imgsz=640,
    ),
}


def weights_path(exp: str) -> Path:

    return RUNS_DIR / exp / "weights" / "best.pt"


# --------------------------------------------------------------------------- data helpers
def raw_image(stem: str) -> Path:
    return RAW_DIR / "images" / f"{stem}.jpg"


def raw_label(stem: str) -> Path:
    return RAW_DIR / "label_txt" / f"{stem}.txt"


def all_stems() -> list[str]:
    return sorted(p.stem for p in (RAW_DIR / "images").glob("*.jpg"))


def read_yolo_labels(path: Path) -> np.ndarray:
    """Read a YOLO label file -> float array (N, 5): class, x_center, y_center, width, height (normalised)."""
    rows = [line.split() for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        return np.zeros((0, 5), dtype=np.float32)
    return np.array(rows, dtype=np.float32)


def yolo_to_xyxy(labels: np.ndarray, w: int = IMG_SIZE, h: int = IMG_SIZE) -> np.ndarray:
    """Normalised (xc, yc, bw, bh) -> pixel (x1, y1, x2, y2)."""
    xc, yc, bw, bh = labels[:, 1] * w, labels[:, 2] * h, labels[:, 3] * w, labels[:, 4] * h
    return np.stack([xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2], axis=1)


def xyxy_to_yolo(boxes: np.ndarray, w: int, h: int) -> np.ndarray:
    """Pixel (x1, y1, x2, y2) -> normalised (xc, yc, bw, bh)."""
    return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2 / w, (boxes[:, 1] + boxes[:, 3]) / 2 / h,
                     (boxes[:, 2] - boxes[:, 0]) / w, (boxes[:, 3] - boxes[:, 1]) / h], axis=1)


def read_split(name: str) -> list[str]:
    """Image stems belonging to a split (train / val / test), as written by 02_prepare_dataset.py."""
    path = SPLITS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run 02_prepare_dataset.py first")
    return [s for s in path.read_text().split() if s]


def load_ground_truth(stem: str) -> tuple[np.ndarray, np.ndarray]:
    """Ground-truth boxes (pixel xyxy) and class ids for one image."""
    lab = read_yolo_labels(raw_label(stem))
    return yolo_to_xyxy(lab), lab[:, 0].astype(int)


DUPLICATE_HASH_DIST = 6  # images whose 256-bit dHash differ in <= 6 bits are treated as near-duplicate frames


def dhash(path: Path, size: int = 16) -> np.ndarray:
    """256-bit difference hash of an image. Nearly identical video frames get (almost) the same hash."""
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_4)
    small = cv2.resize(img, (size + 1, size), interpolation=cv2.INTER_AREA).astype(np.int16)
    return (small[:, 1:] > small[:, :-1]).flatten()


def hash_distances(hashes: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distances between hashes (N x N), diagonal set to a large value."""
    h = hashes.astype(np.float32)  # float32 matmul is exact for these small integers and fast
    dist = h.shape[1] - (h @ h.T + (1 - h) @ (1 - h).T).astype(np.int32)
    np.fill_diagonal(dist, 10_000)
    return dist


# --------------------------------------------------------------------------- plotting
# Categorical colours (fixed order, colour-blind checked) - slot 1 is used for single-series charts.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ_CMAP_COLORS = ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # one-hue blue ramp


def setup_plots():
    """Consistent, print-friendly matplotlib style for report figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#9a9893", "axes.labelcolor": "#2b2a28",
        "xtick.color": "#52514e", "ytick.color": "#52514e",
        "axes.grid": True, "grid.color": "#e4e2dd", "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.axisbelow": True, "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
        "legend.frameon": False,
    })
    return plt


def sequential_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("seq_blue", SEQ_CMAP_COLORS)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
