"""
S5 - Figurer for Faster R-CNN i samme stil som de andre modellene i gruppa:
     1 Originalbilde | 2 Fasit (true labels) | 3 Prediksjoner, med antall per klasse.

Bruker den ferdig trente modellen (best.pt fra S4) og den felles splitten.
Krever matplotlib:  C:\\venvs\\transset\\Scripts\\python.exe -m pip install matplotlib
Kjoer:              C:\\venvs\\transset\\Scripts\\python.exe S5_figurer_faster_rcnn.py
"""
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import torch
import torchvision
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.transforms.functional import to_tensor

# ------------------------- Innstillinger -------------------------
DATA = Path(r"C:\data\TRANSSET_yolo")              # felles split
MODELL = Path(r"C:\trening\faster_rcnn\best.pt")    # fra S4
UT = Path("Faster_RCNN_resultater") / "figurer"
SCORE_TERSKEL = 0.5     # samme som i S4
IOU_TERSKEL = 0.5
BILDESTR = 1024

# Skriv inn de samme bildene som de andre i gruppa brukte, f.eks. ["file_00261"].
# Tom liste = 6 tilfeldige bilder fra testsettet.
BILDER = []
ANTALL_TILFELDIGE = 6
SEED = 42

KLASSER = ["car", "hatchback", "pickup truck", "sedan",
           "suv", "trailer", "truck", "van"]
FARGER = ["#1f6fd1", "#f26b3a", "#2aa876", "#f2a900",
          "#e377c2", "#138a36", "#3b3bb3", "#d62728"]


def finn_split(stem):
    for s in ("train", "val", "test"):
        if (DATA / "images" / s / f"{stem}.jpg").exists():
            return s
    return None


def les_fasit(stem, split, W, H):
    bokser, klasser = [], []
    lbl = DATA / "labels" / split / f"{stem}.txt"
    if lbl.exists():
        for linje in lbl.read_text().splitlines():
            if linje.strip():
                k, x, y, w, h = map(float, linje.split())
                bokser.append([(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H])
                klasser.append(int(k))
    return torch.tensor(bokser, dtype=torch.float32).reshape(-1, 4), torch.tensor(klasser, dtype=torch.int64)


def match(pb, pl, tb, tl):
    """Klasseuavhengig matching (IoU >= terskel), deretter sjekk av klasse."""
    funnet = riktig = 0
    brukt_t = set()
    if len(pb) and len(tb):
        iou = torchvision.ops.box_iou(pb, tb)
        par = sorted(((float(iou[i, j]), i, j) for i in range(len(pb)) for j in range(len(tb))
                      if iou[i, j] >= IOU_TERSKEL), reverse=True)
        brukt_p = set()
        for _, i, j in par:
            if i in brukt_p or j in brukt_t:
                continue
            brukt_p.add(i)
            brukt_t.add(j)
            funnet += 1
            riktig += int(pl[i] == tl[j])
    return funnet, riktig, len(tb) - funnet, len(pb) - funnet


def tegn(ax, bilde, bokser, klasser, scorer=None):
    ax.imshow(bilde)
    for n, (b, k) in enumerate(zip(bokser.tolist(), klasser.tolist())):
        x1, y1, x2, y2 = b
        ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                        edgecolor=FARGER[k], linewidth=1.5))
        tekst = KLASSER[k] if scorer is None else f"{KLASSER[k]} {float(scorer[n]):.2f}"
        ax.text(x1, y1 - 3, tekst, fontsize=6, color="white", weight="bold",
                bbox=dict(facecolor=FARGER[k], edgecolor="none", pad=1))
    ax.axis("off")


def main():
    enhet = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modell = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, num_classes=len(KLASSER) + 1,
                                        min_size=BILDESTR, max_size=BILDESTR, box_detections_per_img=100)
    modell.load_state_dict(torch.load(MODELL, map_location=enhet))
    modell.to(enhet).eval()
    UT.mkdir(parents=True, exist_ok=True)

    bilder = BILDER
    if not bilder:
        alle = sorted(p.stem for p in (DATA / "images" / "test").glob("*.jpg"))
        bilder = random.Random(SEED).sample(alle, ANTALL_TILFELDIGE)

    for stem in bilder:
        split = finn_split(stem)
        if split is None:
            print(f"Fant ikke {stem} i datasettet - hopper over")
            continue
        bilde = Image.open(DATA / "images" / split / f"{stem}.jpg").convert("RGB")
        W, H = bilde.size
        tb, tl = les_fasit(stem, split, W, H)

        with torch.no_grad():
            p = modell([to_tensor(bilde).to(enhet)])[0]
        behold = p["scores"].cpu() >= SCORE_TERSKEL
        pb = p["boxes"].cpu()[behold]
        pl = p["labels"].cpu()[behold] - 1          # tilbake til klasse 0-7
        ps = p["scores"].cpu()[behold]

        funnet, riktig, bom, ekstra = match(pb, pl, tb, tl)

        fig, akser = plt.subplots(1, 3, figsize=(18, 7.2))
        akser[0].imshow(bilde)
        akser[0].axis("off")
        akser[0].set_title("1  Original image", loc="left", fontsize=11, weight="bold")
        tegn(akser[1], bilde, tb, tl)
        akser[1].set_title(f"2  True labels: {len(tb)} vehicles", loc="left", fontsize=11, weight="bold")
        tegn(akser[2], bilde, pb, pl, ps)
        akser[2].set_title(f"3  Faster R-CNN predictions: {len(pb)} vehicles", loc="left",
                           fontsize=11, weight="bold")

        merknad = {"test": "TEST image: not seen during training or model selection",
                   "val": "VALIDATION image: used to choose the best epoch",
                   "train": "TRAINING image: the model learned from it, so expect better results than on new images"}[split]
        fig.text(0.01, 0.985, f"{stem}.jpg", fontsize=13, weight="bold", va="top")
        fig.text(0.01, 0.95, f"Model: Faster R-CNN ResNet-50-FPN v2, full image at {BILDESTR} px, "
                             f"confidence >= {SCORE_TERSKEL}", fontsize=9, va="top")
        fig.text(0.01, 0.925, f"Dataset image {stem}, {merknad}", fontsize=9, va="top")
        fig.text(0.01, 0.90, f"Found {funnet} of {len(tb)} true vehicles (IoU >= {IOU_TERSKEL}), "
                             f"{riktig} with the right class.  Missed: {bom}.  Extra boxes: {ekstra}.",
                 fontsize=9, va="top")

        forklaring = [mpatches.Patch(color=FARGER[k],
                                     label=f"{KLASSER[k]}:  true {int((tl == k).sum())} / pred {int((pl == k).sum())}")
                      for k in range(len(KLASSER))]
        fig.legend(handles=forklaring, loc="lower left", ncol=len(KLASSER), fontsize=8, frameon=False,
                   bbox_to_anchor=(0.005, 0.0))
        fig.subplots_adjust(left=0.005, right=0.995, top=0.85, bottom=0.07, wspace=0.02)
        fil = UT / f"{stem}_faster_rcnn.png"
        fig.savefig(fil, dpi=150)
        plt.close(fig)
        print(f"{stem} ({split}): fant {funnet}/{len(tb)}, riktig klasse {riktig}, "
              f"bom {bom}, ekstra {ekstra}  ->  {fil}")


if __name__ == "__main__":
    main()
