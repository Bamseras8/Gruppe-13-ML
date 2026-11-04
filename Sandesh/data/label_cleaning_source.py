from pathlib import Path
from collections import Counter, defaultdict
import random, shutil, csv

root = Path("TRANSSET_v2")
img_dir = root / "images"
txt_dir = root / "label_txt"
UT = Path(r"C:\data\TRANSSET_yolo")   # ferdig datasett, utenfor OneDrive
TERSKEL = "t8"                         # valgt i steg 3a
SEED = 42
W = H = 1024
MIN_PX = 8
EPS = 1e-6                             # toleranse for avrundingsfeil

klasser = [k.strip() for k in (txt_dir / "classes.txt").read_text().splitlines() if k.strip()]

# --- 1. Les grupper og roterte bilder ---
roterte = set(Path("s2b_roterte_liste.txt").read_text().split())
grupper = defaultdict(list)
with open("s3a_grupper.csv", encoding="utf-8") as f:
    for rad in csv.DictReader(f):
        grupper[rad[TERSKEL]].append(rad["bilde"])

n_tot = len(list(img_dir.glob("*.jpg")))
maal = {"val": round(0.15 * n_tot), "test": round(0.15 * n_tot)}
maal["train"] = n_tot - maal["val"] - maal["test"] - len(roterte)

# --- 2. Fordel grupper: største først, til settet som mangler flest bilder ---
rng = random.Random(SEED)
gruppeliste = list(grupper.values())
rng.shuffle(gruppeliste)
gruppeliste.sort(key=len, reverse=True)
split = {"train": [], "val": [], "test": []}
for g in gruppeliste:
    s = max(maal, key=lambda k: maal[k] - len(split[k]))
    split[s].extend(g)
split["train"].extend(sorted(roterte))   # roterte kun i trening

# Kontroll: ingen bilder i to sett
alle = split["train"] + split["val"] + split["test"]
assert len(alle) == len(set(alle)) == n_tot, "Feil: overlapp eller manglende bilder!"

# --- 3. Rett labels ---
endringer = Counter()

def rett_labels(tekst):
    ut = []
    for linje in tekst.splitlines():
        if not linje.strip():
            continue
        k, x, y, w, h = linje.split()
        x, y, w, h = map(float, (x, y, w, h))
        if w <= 0 or h <= 0:
            endringer["fjernet: bredde/høyde <= 0"] += 1
            continue
        x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
        if x1 < -EPS or y1 < -EPS or x2 > 1 + EPS or y2 > 1 + EPS:
            endringer["klippet til bildekanten"] += 1
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(1.0, x2), min(1.0, y2)
        w, h = x2 - x1, y2 - y1
        if min(w * W, h * H) < MIN_PX:
            endringer[f"fjernet: under {MIN_PX} px"] += 1
            continue
        ut.append((int(k), (x1 + x2) / 2, (y1 + y2) / 2, w, h))
    return ut

# --- 4. Kopier bilder og skriv rettede labels ---
klasse_per_split = {s: Counter() for s in split}
for s, navn in split.items():
    (UT / "images" / s).mkdir(parents=True, exist_ok=True)
    (UT / "labels" / s).mkdir(parents=True, exist_ok=True)
    print(f"Kopierer {s} ({len(navn)} bilder) ...")
    for stem in sorted(navn):
        bokser = rett_labels((txt_dir / f"{stem}.txt").read_text())
        for b in bokser:
            klasse_per_split[s][b[0]] += 1
        (UT / "labels" / s / f"{stem}.txt").write_text(
            "\n".join(f"{k} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for k, x, y, w, h in bokser),
            encoding="utf-8")
        shutil.copy2(img_dir / f"{stem}.jpg", UT / "images" / s / f"{stem}.jpg")
    Path(f"s3_split_{s}.txt").write_text("\n".join(sorted(navn)), encoding="utf-8")

# data.yaml for YOLO / RT-DETR
yaml = (f"path: {UT.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(klasser)))
(UT / "data.yaml").write_text(yaml, encoding="utf-8")

# --- 5. Rapport ---
r = ["=== STEG 3b: SPLIT ===",
     f"Terskel: {TERSKEL}, seed: {SEED}", ""]
for s in split:
    n = len(split[s])
    rot = len([x for x in split[s] if x in roterte])
    r.append(f"{s:<6} {n:>5} bilder ({100 * n / n_tot:.1f} %)  herav roterte: {rot}")
r.append("\nEndringer i labels:")
for k, v in endringer.items():
    r.append(f"  {k}: {v}")
r.append("\nBokser per klasse (andel i train / val / test):")
for kid, navn in enumerate(klasser):
    tot = sum(klasse_per_split[s][kid] for s in split)
    andeler = " / ".join(f"{100 * klasse_per_split[s][kid] / tot:4.1f} %" for s in split)
    r.append(f"  {navn:<14} {tot:>5}   {andeler}")
r.append(f"\nFerdig datasett: {UT}")

tekst = "\n".join(r)
print("\n" + tekst)
Path("s3b_resultat.txt").write_text(tekst, encoding="utf-8")