from pathlib import Path
from collections import Counter

root = Path("TRANSSET_v2")
img_dir = root / "images"
txt_dir = root / "label_txt"


klasser = [k.strip() for k in (txt_dir / "classes.txt").read_text().splitlines() if k.strip()]
gyldige_id = set(range(len(klasser)))


bilder = {p.stem: p for p in img_dir.glob("*.jpg")}
labels = {p.stem: p for p in txt_dir.glob("*.txt") if p.name != "classes.txt"}

bilder_uten_label = sorted(set(bilder) - set(labels))
labels_uten_bilde = sorted(set(labels) - set(bilder))

tomme = []          
feil_format = []    
ugyldig_id = []     
per_klasse = Counter()
bokser_per_bilde = []

for stem in sorted(set(bilder) & set(labels)):
    linjer = [l for l in labels[stem].read_text().splitlines() if l.strip()]
    if not linjer:
        tomme.append(stem)
    antall = 0
    for nr, linje in enumerate(linjer, 1):
        deler = linje.split()
        if len(deler) != 5:
            feil_format.append((stem, nr, linje))
            continue
        try:
            kid = int(deler[0])
            [float(x) for x in deler[1:]]
        except ValueError:
            feil_format.append((stem, nr, linje))
            continue
        if kid not in gyldige_id:
            ugyldig_id.append((stem, nr, kid))
            continue
        per_klasse[kid] += 1
        antall += 1
    bokser_per_bilde.append(antall)


r = []
r.append("=== STEG 1: FILER OG KLASSER ===")
r.append(f"Bilder (.jpg):              {len(bilder)}")
r.append(f"Labelfiler (uten classes):  {len(labels)}")
r.append(f"Bilder uten label:          {len(bilder_uten_label)}  {bilder_uten_label[:10]}")
r.append(f"Labels uten bilde:          {len(labels_uten_bilde)}  {labels_uten_bilde[:10]}")
r.append(f"Tomme labelfiler:           {len(tomme)}  {tomme[:10]}")
r.append(f"Linjer med feil format:     {len(feil_format)}  {feil_format[:5]}")
r.append(f"Ugyldige klasse-ID-er:      {len(ugyldig_id)}  {ugyldig_id[:5]}")
r.append("")
r.append("Antall bokser per klasse:")
for kid, navn in enumerate(klasser):
    r.append(f"  {kid}  {navn:<14} {per_klasse[kid]}")
r.append(f"  Totalt              {sum(per_klasse.values())}")
r.append("")
r.append(f"Bokser per bilde: min {min(bokser_per_bilde)}, "
         f"maks {max(bokser_per_bilde)}, "
         f"snitt {sum(bokser_per_bilde)/len(bokser_per_bilde):.1f}")

tekst = "\n".join(r)
print(tekst)
Path("steg1_resultat.txt").write_text(tekst, encoding="utf-8")
print("\nLagret til steg1_resultat.txt")