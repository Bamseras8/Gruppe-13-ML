from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
import imagehash

root = Path("TRANSSET_v2")
img_dir = root / "images"
roterte = set(Path("s2b_roterte_liste.txt").read_text().split())

bilder = sorted(p for p in img_dir.glob("*.jpg") if p.stem not in roterte)
print(f"Beregner fingeravtrykk for {len(bilder)} uroterte bilder ...")

hasher = []
for i, b in enumerate(bilder):
    with Image.open(b) as im:
        hasher.append(imagehash.phash(im).hash.flatten())
    if (i + 1) % 500 == 0:
        print(f"  {i + 1}/{len(bilder)}")
H = np.array(hasher, dtype=np.uint8)   # ett 64-bits fingeravtrykk per bilde


def grupper(H, terskel):
    """Bilder med avstand <= terskel kobles; sammenkoblede bilder = én gruppe."""
    n = len(H)
    forelder = list(range(n))

    def finn(a):
        while forelder[a] != a:
            forelder[a] = forelder[forelder[a]]
            a = forelder[a]
        return a

    for i in range(n):
        avstand = (H[i + 1:] != H[i]).sum(axis=1)
        for j in np.nonzero(avstand <= terskel)[0] + i + 1:
            ri, rj = finn(i), finn(int(j))
            if ri != rj:
                forelder[ri] = rj
    return [finn(i) for i in range(n)]


terskler = [4, 8, 12, 16]
resultater = {}
r = ["=== STEG 3a: GRUPPERING AV LIKE BILDER ===",
     f"Uroterte bilder: {len(bilder)}", ""]
for t in terskler:
    g = grupper(H, t)
    resultater[t] = g
    str_ = Counter(g)
    storst = max(str_.values())
    alene = sum(1 for v in str_.values() if v == 1)
    r.append(f"Terskel {t:>2}: {len(str_):>5} grupper | største gruppe {storst:>4} bilder "
             f"({100 * storst / len(bilder):.1f} %) | {alene} bilder alene")

# Eksempler på grupper ved terskel 8, til visuell kontroll
r.append("\nEksempelgrupper (terskel 8) - åpne noen og se om de ligner:")
str8 = Counter(resultater[8])
vist = 0
for gid, antall in str8.most_common():
    if antall >= 3 and vist < 3:
        navn = [bilder[i].stem for i, g in enumerate(resultater[8]) if g == gid][:6]
        r.append(f"  Gruppe med {antall} bilder: {navn}")
        vist += 1

tekst = "\n".join(r)
print("\n" + tekst)
Path("s3a_resultat.txt").write_text(tekst, encoding="utf-8")

# Lagre gruppene for alle terskler (brukes i steg 3b)
linjer = ["bilde," + ",".join(f"t{t}" for t in terskler)]
for i, b in enumerate(bilder):
    linjer.append(b.stem + "," + ",".join(str(resultater[t][i]) for t in terskler))
Path("s3a_grupper.csv").write_text("\n".join(linjer), encoding="utf-8")
print("\nLagret s3a_resultat.txt og s3a_grupper.csv")