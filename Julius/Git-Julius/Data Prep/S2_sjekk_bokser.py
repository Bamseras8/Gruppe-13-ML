from pathlib import Path
from collections import Counter
from statistics import median
from PIL import Image

root = Path("TRANSSET_v2")
img_dir = root / "images"
txt_dir = root / "label_txt"
EPS = 1e-6          # liten toleranse for avrundingsfeil
SMA_PX = 8          # grense for "svært liten" boks i piksler

storrelser = Counter()
null_str = []       # bredde/høyde <= 0
utenfor = []        # boks går utenfor bildet
duplikater = []     # identiske linjer i samme fil
sma = []            # bokser under SMA_PX
bredder, hoyder = [], []

for lbl in sorted(txt_dir.glob("*.txt")):
    if lbl.name == "classes.txt":
        continue
    with Image.open(img_dir / (lbl.stem + ".jpg")) as im:
        W, H = im.size
    storrelser[(W, H)] += 1

    linjer = [l.strip() for l in lbl.read_text().splitlines() if l.strip()]
    for linje, n in Counter(linjer).items():
        if n > 1:
            duplikater.append((lbl.stem, linje, n))

    for nr, linje in enumerate(linjer, 1):
        k, x, y, w, h = linje.split()
        x, y, w, h = map(float, (x, y, w, h))
        if w <= 0 or h <= 0:
            null_str.append((lbl.stem, nr))
            continue
        if (x - w/2 < -EPS or x + w/2 > 1 + EPS or
                y - h/2 < -EPS or y + h/2 > 1 + EPS):
            utenfor.append((lbl.stem, nr))
        bw, bh = w * W, h * H
        bredder.append(bw)
        hoyder.append(bh)
        if min(bw, bh) < SMA_PX:
            sma.append((lbl.stem, nr, round(bw), round(bh)))

# Rapport
r = []
r.append("=== STEG 2: BOUNDING-BOKSER ===")
r.append("Bildestørrelser (bredde x høyde: antall):")
for (W, H), n in storrelser.most_common():
    r.append(f"  {W} x {H}: {n}")
r.append(f"Bokser sjekket:              {len(bredder) + len(null_str)}")
r.append(f"Bredde/høyde <= 0:           {len(null_str)}  {null_str[:5]}")
r.append(f"Går utenfor bildekanten:     {len(utenfor)}  {utenfor[:5]}")
r.append(f"Duplikatlinjer:              {len(duplikater)}  {duplikater[:5]}")
r.append(f"Svært små bokser (<{SMA_PX} px): {len(sma)}  {sma[:5]}")
r.append("")
r.append("Boksstørrelse i piksler (min / median / maks):")
r.append(f"  Bredde: {min(bredder):.0f} / {median(bredder):.0f} / {max(bredder):.0f}")
r.append(f"  Høyde:  {min(hoyder):.0f} / {median(hoyder):.0f} / {max(hoyder):.0f}")

tekst = "\n".join(r)
print(tekst)
Path("s2_resultat.txt").write_text(tekst, encoding="utf-8")
print("\nLagret til s2_resultat.txt")