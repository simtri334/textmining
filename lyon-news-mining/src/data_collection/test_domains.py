"""Étape 1 : teste la couverture GDELT de chaque média lyonnais, domaine par domaine.

Usage : python -m src.data_collection.test_domains   (depuis la racine du projet)
Produit : data/raw/test_<domaine>_<horodatage>.json (réponses brutes)
          outputs/tables/step1_coverage.csv (tableau de synthèse)
"""
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import gdelt_client as gdelt

ROOT = Path(__file__).resolve().parents[2]
RAW, TABLES = ROOT / "data" / "raw", ROOT / "outputs" / "tables"

# nom du média -> (requête GDELT). Les requêtes diffèrent seulement quand le domaine est partagé.
MEDIA = {
    "Témoin : Le Monde": "domain:lemonde.fr",   # contrôle testé en premier : valide que la méthode renvoie bien des résultats
    "Le Progrès": "domain:leprogres.fr",
    "Lyon Capitale": "domain:lyoncapitale.fr",
    "Rue89Lyon": "domain:rue89lyon.fr",
    "LyonMag": "domain:lyonmag.com",
    "BFM Lyon": "domain:bfm-lyon.com",
    "Tribune de Lyon": "domain:tribunedelyon.fr",
    "Actu (tout le réseau)": "domain:actu.fr",
    "Actu Lyon (sous-domaine)": "domain:lyon.actu.fr",
}


def summarize(name, search, articles, true_count):
    df = pd.DataFrame(articles)
    row = {"Média": name, "Requête": search, "Articles (artlist, max 250)": len(df),
           "Articles réels (timelinevolraw, 3 mois)": true_count}
    if df.empty:
        return {**row, "Première date": None, "Dernière date": None, "Jours distincts": 0,
                "URL en double": 0, "Titres en double": 0, "Langues": "", "Pays": "", "Domaines renvoyés": ""}
    dates = pd.to_datetime(df["seendate"], format="%Y%m%dT%H%M%SZ", errors="coerce")
    return {**row,
            "Première date": dates.min(), "Dernière date": dates.max(),
            "Jours distincts": dates.dt.date.nunique(),
            "URL en double": int(df["url"].duplicated().sum()),
            "Titres en double": int(df["title"].duplicated().sum()),
            "Langues": ", ".join(f"{k}:{v}" for k, v in df["language"].value_counts().items()),
            "Pays": ", ".join(f"{k}:{v}" for k, v in df["sourcecountry"].value_counts().items()),
            "Domaines renvoyés": ", ".join(f"{k}:{v}" for k, v in df["domain"].value_counts().head(3).items())}


def main():
    RAW.mkdir(parents=True, exist_ok=True); TABLES.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = []
    for name, search in MEDIA.items():
        print(f"→ {name} ({search})")
        try:
            articles = gdelt.artlist(search)
            volume = gdelt.timeline_volume(search)
        except RuntimeError as e:      # on garde une trace de l'échec au lieu de perdre toute la série
            print(f"   ERREUR : {e}")
            rows.append({"Média": name, "Requête": search, "Erreur": str(e)[-60:]})
            continue
        true_count = int(sum(p["value"] for p in volume))
        (RAW / f"test_{search.split(':')[1]}_{stamp}.json").write_text(
            json.dumps({"query": search, "articles": articles, "timeline": volume}, ensure_ascii=False))
        rows.append(summarize(name, search, articles, true_count))
        print(f"   {len(articles)} articles renvoyés, {true_count} au total sur 3 mois")
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "step1_coverage.csv", index=False)
    print(out.drop(columns=["Requête", "Langues", "Pays"]).to_string(index=False))


if __name__ == "__main__":
    main()
