"""Collecte des articles de médias lyonnais à partir des fichiers bruts GDELT GKG 2.0.

Pourquoi le GKG : l'API DOC 2.0 était bloquée (HTTP 429 / réponses vides) ; les fichiers GKG,
publiés toutes les 15 min sur data.gdeltproject.org, contiennent les mêmes articles sans limite de requêtes.

Principe : pour chaque tranche de 15 min, on télécharge le zip EN MÉMOIRE, on ne garde que les lignes
dont l'hôte de l'URL appartient à un média ciblé, et on n'écrit sur disque que ces lignes (1 CSV gzip par jour).
Rien d'autre n'est conservé, donc le corpus reste léger. La collecte est reprenable : un jour déjà écrit est sauté.

Usage (depuis la racine du projet) :
    python -m src.data_collection.gkg_collector 2026-11-01 2026-11-01      # un jour
    python -m src.data_collection.gkg_collector 2026-08-03 2026-11-01      # une période
"""
import csv
import gzip
import html
import io
import re
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "raw" / "gkg"
# Flux "translation" : médias non anglophones (traduits automatiquement par GDELT). Le flux GKG standard est
# quasi exclusivement anglophone (mesuré le 2026-11-01 : 7 articles en .fr sur 54 361, aucun média lyonnais).
BASE = "https://data.gdeltproject.org/gdeltv2/{ts}.translation.gkg.csv.zip"

# média -> suffixes d'hôte acceptés (comparés à l'hôte de l'URL, pas à SourceCommonName, plus ambigu)
MEDIA = {
    "Le Progrès": ["leprogres.fr"],
    "Lyon Capitale": ["lyoncapitale.fr"],
    "Rue89Lyon": ["rue89lyon.fr"],
    "LyonMag": ["lyonmag.com"],
    "BFM Lyon": ["bfmtv.com/lyon", "bfm-lyon.com"],   # bfmtv.com/lyon traité à part ci-dessous (chemin d'URL)
    "Tribune de Lyon": ["tribunedelyon.fr"],
    "Actu Lyon": ["lyon.actu.fr"],
}
HOST_TO_MEDIA = {h: m for m, hosts in MEDIA.items() for h in hosts if "/" not in h}

# Index des colonnes du GKG 2.0 (fichier tabulé sans en-tête, 27 colonnes)
COL = {"record_id": 0, "date": 1, "domain": 3, "url": 4, "themes": 8, "locations": 10, "persons": 12,
       "organizations": 14, "tone": 15, "sharing_image": 18, "translation_info": 25, "extras": 26}
FIELDS = ["record_id", "date", "media_name", "domain", "url", "title", "themes", "locations", "persons",
          "organizations", "tone", "sharing_image", "translation_info"]
TITLE_RE = re.compile(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", re.S)

csv.field_size_limit(sys.maxsize)


def match_media(url: str):
    """Renvoie le nom du média si l'URL appartient à l'un des médias ciblés, sinon None."""
    p = urlparse(url)
    host = (p.hostname or "").lower()
    for suffix, media in HOST_TO_MEDIA.items():
        if host == suffix or host.endswith("." + suffix):
            return media
    if host.endswith("bfmtv.com") and p.path.startswith("/lyon"):
        return "BFM Lyon"
    return None


def fetch_slot(ts: str, retries: int = 4):
    """Télécharge et filtre une tranche de 15 min. Renvoie (liste de lignes, nombre total de lignes lues)."""
    for attempt in range(retries):
        try:
            r = requests.get(BASE.format(ts=ts), timeout=90)
            if r.status_code == 404:
                return [], -1                  # tranche absente chez GDELT (trous réels possibles) : tracée à part
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                text = io.TextIOWrapper(z.open(z.namelist()[0]), encoding="utf-8", errors="replace", newline="")
                kept, total = [], 0
                for row in csv.reader(text, delimiter="\t", quoting=csv.QUOTE_NONE):
                    total += 1
                    if len(row) < 27:
                        continue
                    media = match_media(row[COL["url"]])
                    if media:
                        m = TITLE_RE.search(row[COL["extras"]])
                        rec = {k: row[i] for k, i in COL.items() if k != "extras"}
                        # GDELT encode les accents du titre en entités HTML (&#xE9;) : on les décode
                        rec.update(media_name=media, title=html.unescape(m.group(1)).strip() if m else "")
                        kept.append(rec)
                return kept, total
        except (requests.RequestException, zipfile.BadZipFile):
            if attempt == retries - 1:
                raise
    return [], 0


def collect_day(day: date, workers: int = 8):
    out = OUT_DIR / f"{day:%Y%m%d}.csv.gz"
    if out.exists():
        print(f"{day} déjà collecté, ignoré")
        return
    slots = [f"{day:%Y%m%d}{h:02d}{m:02d}00" for h in range(24) for m in (0, 15, 30, 45)]
    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(fetch_slot, slots))
    rows = [r for kept, _ in results for r in kept]
    missing = [s for s, (_, t) in zip(slots, results) if t == -1]
    scanned = sum(t for _, t in results if t > 0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(out.with_suffix(".tmp"), "wt", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader(); w.writerows(rows)
    out.with_suffix(".tmp").rename(out)       # écriture atomique : un fichier incomplet n'est jamais pris pour un jour fini
    print(f"{day} : {scanned:,} articles GKG parcourus, {len(rows)} retenus, {len(missing)} tranches absentes")


def main():
    start, end = (datetime.strptime(a, "%Y-%m-%d").date() for a in sys.argv[1:3])
    d = start
    while d <= end:
        collect_day(d)
        d += timedelta(days=1)


if __name__ == "__main__":
    main()
