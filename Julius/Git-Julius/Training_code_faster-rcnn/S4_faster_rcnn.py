"""
S4 - Faster R-CNN (ResNet50-FPN v2) paa TRANSSET: trening + evaluering

Bruker den felles splitten i C:\\data\\TRANSSET_yolo (laget av S3b_lag_split.py).
Trener paa train, velger beste epoke paa val (mAP@0.5), og evaluerer EN gang paa test.

Kjoer foerst med RASK_TEST = True for aa sjekke at alt virker (noen minutter),
deretter med RASK_TEST = False for full trening.
"""
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection import MeanAveragePrecision
from torchvision.models.detection import (FasterRCNN_ResNet50_FPN_V2_Weights,
                                          fasterrcnn_resnet50_fpn_v2)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import to_tensor

# ------------------------- Innstillinger -------------------------
DATA = Path(r"C:\data\TRANSSET_yolo")      # felles split fra S3b
UT = Path(r"C:\trening\faster_rcnn")        # resultater (utenfor OneDrive)
RASK_TEST = False    # True: 100 treningsbilder, 1 epoke. Sett False for full trening.
EPOKER = 10          # full trening
BATCH = 2            # senk til 1 hvis du faar "CUDA out of memory"
BILDESTR = 1024      # 800 gir raskere trening, men litt daarligere paa smaa kjoeretoey
LR = 0.005
SCORE_TERSKEL = 0.5  # brukes for precision/recall/F1 og telling (ikke for mAP)
IOU_TERSKEL = 0.5
SEED = 42
ANTALL_EKSEMPELBILDER = 6

KLASSER = ["car", "hatchback", "pickup truck", "sedan",
           "suv", "trailer", "truck", "van"]      # samme rekkefoelge som classes.txt
# Faster R-CNN bruker klasse 0 som bakgrunn, saa YOLO-klasse k blir k + 1 her.


# ------------------------- Datasett -------------------------
class YoloDatasett(Dataset):
    """Leser bilder og YOLO-labels (klasse x y b h, normalisert) fra den felles splitten."""

    def __init__(self, split, flip=False, maks=None):
        self.bilder = sorted((DATA / "images" / split).glob("*.jpg"))
        if maks:
            self.bilder = self.bilder[:maks]
        self.lbl_dir = DATA / "labels" / split
        self.flip = flip

    def __len__(self):
        return len(self.bilder)

    def __getitem__(self, i):
        sti = self.bilder[i]
        bilde = Image.open(sti).convert("RGB")
        W, H = bilde.size
        bokser, klasser = [], []
        lbl = self.lbl_dir / (sti.stem + ".txt")
        if lbl.exists():
            for linje in lbl.read_text().splitlines():
                if not linje.strip():
                    continue
                k, x, y, w, h = map(float, linje.split())
                bokser.append([(x - w / 2) * W, (y - h / 2) * H,
                               (x + w / 2) * W, (y + h / 2) * H])
                klasser.append(int(k) + 1)
        bokser = torch.tensor(bokser, dtype=torch.float32).reshape(-1, 4)
        klasser = torch.tensor(klasser, dtype=torch.int64)
        bilde = to_tensor(bilde)
        if self.flip and random.random() < 0.5:          # enkel augmentering
            bilde = bilde.flip(-1)
            if len(bokser):
                bokser[:, [0, 2]] = W - bokser[:, [2, 0]]
        return bilde, {"boxes": bokser, "labels": klasser}, sti.stem


def samle(batch):
    return tuple(zip(*batch))


# ------------------------- Modell -------------------------
def lag_modell():
    modell = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,   # forhaandstrent paa COCO
        min_size=BILDESTR, max_size=BILDESTR,
        box_detections_per_img=100)
    inn = modell.roi_heads.box_predictor.cls_score.in_features
    modell.roi_heads.box_predictor = FastRCNNPredictor(inn, len(KLASSER) + 1)
    return modell


@torch.no_grad()
def prediker(modell, loader, enhet):
    modell.eval()
    pred, maal, navn = [], [], []
    for bilder, t, n in loader:
        bilder = [b.to(enhet) for b in bilder]
        with torch.autocast(device_type=enhet.type, enabled=enhet.type == "cuda"):
            ut = modell(bilder)
        for p in ut:
            pred.append({"boxes": p["boxes"].float().cpu(),
                         "scores": p["scores"].float().cpu(),
                         "labels": p["labels"].cpu()})
        maal += list(t)
        navn += list(n)
    return pred, maal, navn


# ------------------------- Evaluering -------------------------
def beregn_map(pred, maal):
    alle = MeanAveragePrecision(iou_type="bbox")
    alle.update(pred, maal)
    r = alle.compute()
    ved50 = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.5], class_metrics=True)
    ved50.update(pred, maal)
    r50 = ved50.compute()
    per_klasse = {}
    klasser = r50["classes"].reshape(-1).tolist()
    verdier = r50["map_per_class"].reshape(-1).tolist()
    for c, v in zip(klasser, verdier):
        per_klasse[KLASSER[int(c) - 1]] = float(v)
    return {"mAP@0.5": float(r["map_50"]),
            "mAP@0.5:0.95": float(r["map"]),
            "AP@0.5 per klasse": per_klasse}


def precision_recall_f1(pred, maal):
    teller = {c: [0, 0, 0] for c in range(1, len(KLASSER) + 1)}   # TP, FP, FN
    for p, t in zip(pred, maal):
        behold = p["scores"] >= SCORE_TERSKEL
        pb, pl, ps = p["boxes"][behold], p["labels"][behold], p["scores"][behold]
        tb, tl = t["boxes"], t["labels"]
        brukt = torch.zeros(len(tb), dtype=torch.bool)
        iou = torchvision.ops.box_iou(pb, tb) if len(pb) and len(tb) else None
        for i in torch.argsort(ps, descending=True).tolist():
            c = int(pl[i])
            if iou is not None:
                kandidat = iou[i].clone()
                kandidat[brukt | (tl != c)] = -1
                j = int(torch.argmax(kandidat))
                if kandidat[j] >= IOU_TERSKEL:
                    brukt[j] = True
                    teller[c][0] += 1
                    continue
            teller[c][1] += 1
        for j in range(len(tb)):
            if not brukt[j]:
                teller[int(tl[j])][2] += 1

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return {"precision": p, "recall": r, "F1": f, "TP": tp, "FP": fp, "FN": fn}

    tot = [sum(v[i] for v in teller.values()) for i in range(3)]
    return {"totalt": prf(*tot),
            "per klasse": {KLASSER[c - 1]: prf(*v) for c, v in teller.items()}}


def telling(pred, maal):
    feil_tot = []
    feil_kl = {c: [] for c in range(1, len(KLASSER) + 1)}
    for p, t in zip(pred, maal):
        pl = p["labels"][p["scores"] >= SCORE_TERSKEL]
        feil_tot.append(len(pl) - len(t["labels"]))
        for c in feil_kl:
            feil_kl[c].append(int((pl == c).sum()) - int((t["labels"] == c).sum()))

    def mae_rmse(e):
        e = np.array(e, dtype=float)
        return {"MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2)))}

    return {"totalt": mae_rmse(feil_tot),
            "per klasse": {KLASSER[c - 1]: mae_rmse(v) for c, v in feil_kl.items()}}


def tegn_eksempler(datasett, pred, navn, mappe):
    mappe.mkdir(parents=True, exist_ok=True)
    indekser = list(range(len(navn)))
    random.Random(SEED).shuffle(indekser)
    for i in indekser[:ANTALL_EKSEMPELBILDER]:
        bilde = Image.open(datasett.bilder[i]).convert("RGB")
        d = ImageDraw.Draw(bilde)
        p = pred[i]
        for b, l, s in zip(p["boxes"], p["labels"], p["scores"]):
            if s < SCORE_TERSKEL:
                continue
            x1, y1, x2, y2 = b.tolist()
            d.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
            d.text((x1 + 3, y1 + 2), f"{KLASSER[int(l) - 1]} {s:.2f}", fill=(255, 255, 0))
        bilde.save(mappe / f"{navn[i]}_prediksjon.jpg")


# ------------------------- Hovedprogram -------------------------
def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    enhet = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Enhet:", torch.cuda.get_device_name(0) if enhet.type == "cuda" else "CPU (ADVARSEL: veldig tregt)")
    UT.mkdir(parents=True, exist_ok=True)

    epoker = 1 if RASK_TEST else EPOKER
    tren = YoloDatasett("train", flip=True, maks=100 if RASK_TEST else None)
    val = YoloDatasett("val", maks=50 if RASK_TEST else None)
    test = YoloDatasett("test", maks=50 if RASK_TEST else None)
    print(f"Bilder: train {len(tren)}, val {len(val)}, test {len(test)}"
          + ("   [RASK_TEST]" if RASK_TEST else ""))

    kw = dict(batch_size=BATCH, num_workers=4, collate_fn=samle, pin_memory=enhet.type == "cuda")
    tren_l = DataLoader(tren, shuffle=True, **kw)
    val_l = DataLoader(val, shuffle=False, **kw)
    test_l = DataLoader(test, shuffle=False, **kw)

    modell = lag_modell().to(enhet)
    parametere = [p for p in modell.parameters() if p.requires_grad]
    opt = torch.optim.SGD(parametere, lr=LR, momentum=0.9, weight_decay=1e-4)
    steg_totalt = epoker * len(tren_l)
    oppvarming = min(300, len(tren_l))

    def lr_faktor(s):   # lineaer oppvarming, deretter cosinus-nedtrapping
        if s < oppvarming:
            return (s + 1) / oppvarming
        return 0.5 * (1 + math.cos(math.pi * (s - oppvarming) / max(1, steg_totalt - oppvarming)))

    planlegger = torch.optim.lr_scheduler.LambdaLR(opt, lr_faktor)
    skalerer = torch.amp.GradScaler("cuda", enabled=enhet.type == "cuda")

    logg = []
    beste = -1.0
    start = time.time()
    for ep in range(1, epoker + 1):
        modell.train()
        t0 = time.time()
        sum_tap = 0.0
        for i, (bilder, maal, _) in enumerate(tren_l, 1):
            bilder = [b.to(enhet) for b in bilder]
            maal = [{k: v.to(enhet) for k, v in t.items()} for t in maal]
            with torch.autocast(device_type=enhet.type, enabled=enhet.type == "cuda"):
                tap = sum(modell(bilder, maal).values())
            opt.zero_grad()
            skalerer.scale(tap).backward()
            skalerer.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(parametere, 10.0)
            skalerer.step(opt)
            skalerer.update()
            planlegger.step()
            sum_tap += tap.item()
            if i % 100 == 0:
                print(f"  epoke {ep} steg {i}/{len(tren_l)}  loss {sum_tap / i:.3f}")
        snitt_tap = sum_tap / len(tren_l)

        p, t, _ = prediker(modell, val_l, enhet)
        r = beregn_map(p, t)
        logg.append({"epoke": ep, "train_loss": snitt_tap,
                     "val_mAP50": r["mAP@0.5"], "val_mAP50_95": r["mAP@0.5:0.95"],
                     "minutter": (time.time() - t0) / 60})
        print(f"Epoke {ep}/{epoker}: loss {snitt_tap:.3f} | val mAP@0.5 {r['mAP@0.5']:.3f} "
              f"| val mAP@0.5:0.95 {r['mAP@0.5:0.95']:.3f} | {(time.time() - t0) / 60:.1f} min")
        if r["mAP@0.5"] > beste:
            beste = r["mAP@0.5"]
            torch.save(modell.state_dict(), UT / "best.pt")
            print("  -> ny beste modell lagret")

    with open(UT / "treningslogg.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(logg[0].keys()))
        w.writeheader()
        w.writerows(logg)

    # ---------- Endelig evaluering paa testsettet (beste modell) ----------
    print("\nEvaluerer beste modell paa testsettet ...")
    modell.load_state_dict(torch.load(UT / "best.pt", map_location=enhet))
    p, t, navn = prediker(modell, test_l, enhet)
    resultat = {
        "innstillinger": {"epoker": epoker, "batch": BATCH, "bildestr": BILDESTR, "lr": LR,
                          "score_terskel": SCORE_TERSKEL, "iou_terskel": IOU_TERSKEL,
                          "rask_test": RASK_TEST, "treningstid_min": (time.time() - start) / 60},
        "mAP": beregn_map(p, t),
        "deteksjon": precision_recall_f1(p, t),
        "telling": telling(p, t),
    }
    (UT / "test_resultater.json").write_text(json.dumps(resultat, indent=2, ensure_ascii=False),
                                             encoding="utf-8")
    tegn_eksempler(test, p, navn, UT / "eksempler")

    d, m, c = resultat["deteksjon"], resultat["mAP"], resultat["telling"]
    r = ["=== FASTER R-CNN - TESTRESULTATER ===",
         f"Testbilder: {len(test)}   Epoker: {epoker}   Bildestr: {BILDESTR}"
         + ("   [RASK_TEST - ikke bruk i rapporten]" if RASK_TEST else ""),
         "",
         f"mAP@0.5:       {m['mAP@0.5']:.3f}",
         f"mAP@0.5:0.95:  {m['mAP@0.5:0.95']:.3f}",
         f"Precision:     {d['totalt']['precision']:.3f}",
         f"Recall:        {d['totalt']['recall']:.3f}",
         f"F1-score:      {d['totalt']['F1']:.3f}",
         f"Telling totalt: MAE {c['totalt']['MAE']:.2f}   RMSE {c['totalt']['RMSE']:.2f}",
         "",
         f"{'Klasse':<14}{'AP@0.5':>8}{'Prec':>8}{'Recall':>8}{'F1':>8}{'MAE':>8}{'RMSE':>8}"]
    for k in KLASSER:
        ap = m["AP@0.5 per klasse"].get(k, float("nan"))
        pk, ck = d["per klasse"][k], c["per klasse"][k]
        r.append(f"{k:<14}{ap:>8.3f}{pk['precision']:>8.3f}{pk['recall']:>8.3f}"
                 f"{pk['F1']:>8.3f}{ck['MAE']:>8.2f}{ck['RMSE']:>8.2f}")
    r.append(f"\nLagret i: {UT}")
    tekst = "\n".join(r)
    print("\n" + tekst)
    (UT / "test_resultater.txt").write_text(tekst, encoding="utf-8")


if __name__ == "__main__":   # nødvendig paa Windows naar DataLoader bruker flere prosesser
    main()