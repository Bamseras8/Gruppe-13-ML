from pathlib import Path
from PIL import Image

img_dir = Path("TRANSSET_v2") / "images"
GRENSE = 30   # pikselsum under dette regnes som svart

def er_rotert(sti):
    with Image.open(sti) as im:
        im = im.convert("RGB")
        W, H = im.size
        hjorner = [(2, 2), (W - 3, 2), (2, H - 3), (W - 3, H - 3)]
        svarte = sum(1 for p in hjorner if sum(im.getpixel(p)) < GRENSE)
    return svarte >= 2   # minst to svarte hjørner

bilder = sorted(img_dir.glob("*.jpg"))
roterte = [b.stem for b in bilder if er_rotert(b)]

tekst = (f"=== STEG 2b: ROTERTE BILDER ===\n"
         f"Bilder totalt:  {len(bilder)}\n"
         f"Rotert (svarte hjørner): {len(roterte)} "
         f"({100 * len(roterte) / len(bilder):.1f} %)\n"
         f"Eksempler: {roterte[:15]}")
print(tekst)
Path("s2b_resultat.txt").write_text(tekst, encoding="utf-8")
Path("s2b_roterte_liste.txt").write_text("\n".join(roterte), encoding="utf-8")