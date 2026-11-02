"""Étape 2 : construit le corpus propre à partir des fichiers bruts GKG (data/raw/gkg/*.csv.gz).

Les fichiers bruts ne sont jamais modifiés. Sorties :
    data/processed/corpus.parquet            corpus dédoublonné (une ligne = un article)
    outputs/tables/step2_coverage.csv        tableau de couverture par média
Usage : python -m src.preprocessing.build_corpus
"""
import glob
import hashlib
import re
from urllib.parse import urlparse

import pandas as pd

from .lyon_filter import lyon_flag

ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
RAW, OUT, TABLES = ROOT / "data" / "raw" / "gkg", ROOT / "data" / "processed", ROOT / "outputs" / "tables"

URL_DATE = re.compile(r"/(20\d{2})/(\d{2})/(\d{2})/")
MAX_URL_GAP_DAYS = 2      # au-delà, la date de l'URL est celle d'un événement (agenda) et non de la publication
LYON_BASED_MEDIA = ["Lyon Capitale", "LyonMag", "BFM Lyon"]
PLACE_PREFIX = re.compile(r"^([^.]{2,40})\. ")   # titres du Progrès : « Lagnieu. Un homme… »


def normalize_url(url: str) -> str:
    """Clé de dédoublonnage : schéma https, hôte minuscule, sans query, ancre ni slash final."""
    p = urlparse(url.strip())
    return f"https://{(p.hostname or '').lower()}{p.path.rstrip('/')}"


def extract_section(url: str, media: str):
    """Rubrique éditoriale lue dans le chemin de l'URL (None si le média n'en a pas)."""
    parts = [s for s in urlparse(url).path.split("/") if s]
    if media == "BFM Lyon":                      # bfmtv.com/lyon/<rubrique>/<slug> ou bfmtv.com/lyon/<slug>
        return parts[1] if len(parts) > 2 else None
    if media in ("Le Progrès", "Lyon Capitale"):
        return parts[0] if len(parts) > 1 else None
    return None                                   # LyonMag : /article/<id>/<slug>, pas de rubrique


def flag_low_coverage_days(df: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
    """Marque les articles d'un (média, jour) anormalement creux : moins de `threshold` × la médiane du MÊME jour
    de la semaine sur ±3 semaines. Tenir compte du jour de la semaine évite de prendre un week-end pour une panne.
    Seuls les médias à volume suffisant (> 5 articles/jour en médiane) sont évalués."""
    daily = df.groupby(["media_name", "date"]).size().rename("n").reset_index()
    flagged = set()
    for media, g in daily.groupby("media_name"):
        g = g.set_index("date")["n"].reindex(pd.date_range(df["date"].min(), df["date"].max()), fill_value=0)
        if g.median() <= 5:
            continue
        for d in g.index:
            peers = [g.get(d + pd.Timedelta(weeks=k)) for k in (-3, -2, -1, 1, 2, 3)]
            peers = [x for x in peers if x is not None]
            if len(peers) >= 3 and g[d] < threshold * pd.Series(peers).median():
                flagged.add((media, d))
    return pd.Series([(m, d) in flagged for m, d in zip(df["media_name"], df["date"])], index=df.index)


def load_raw() -> pd.DataFrame:
    files = sorted(glob.glob(str(RAW / "*.csv.gz")))
    return pd.concat((pd.read_csv(f, dtype={"date": str}) for f in files), ignore_index=True)


def build() -> tuple[pd.DataFrame, dict]:
    raw = load_raw()
    stats = {"lignes_brutes": len(raw)}

    raw["url_key"] = raw["url"].map(normalize_url)
    df = raw.sort_values("date").drop_duplicates("url_key", keep="first").copy()   # on garde la 1re détection
    stats["doublons_url_supprimes"] = len(raw) - len(df)

    df["article_id"] = df["url_key"].map(lambda u: hashlib.sha1(u.encode()).hexdigest()[:12])
    df["seendate"] = pd.to_datetime(df["date"], format="%Y%m%d%H%M%S")
    df["title"] = df["title"].fillna("").str.strip()

    # Date de publication : date de l'URL si proche de la détection GDELT, sinon date de détection
    m = df["url"].str.extract(URL_DATE)
    url_date = pd.to_datetime(m[0] + "-" + m[1] + "-" + m[2], errors="coerce")
    gap = (df["seendate"].dt.normalize() - url_date).dt.days.abs()
    use_url = url_date.notna() & (gap <= MAX_URL_GAP_DAYS)
    df["date"] = url_date.where(use_url, df["seendate"].dt.normalize())
    df["date_source"] = use_url.map({True: "url", False: "gdelt_seendate"})
    df["year"], df["month"] = df["date"].dt.year, df["date"].dt.month
    iso = df["date"].dt.isocalendar()
    df["week"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    df["week_start"] = df["date"] - pd.to_timedelta(df["date"].dt.day_of_week, unit="D")
    df["day_of_week"] = df["date"].dt.day_of_week          # 0 = lundi

    # Langue d'origine (champ GDELT « srclc:fra;eng:GT-FRA… ») : fra = français
    df["language"] = df["translation_info"].str.extract(r"srclc:(\w+)")[0]

    # Rubrique, préfixe de lieu, titre sans préfixe
    df["section"] = [extract_section(u, mm) for u, mm in zip(df["url"], df["media_name"])]
    prefix = df["title"].str.extract(PLACE_PREFIX)[0].str.strip()
    df["place_prefix"] = prefix.where(df["media_name"] == "Le Progrès")
    df["title_clean"] = df["title"].where(df["place_prefix"].isna(),
                                          df["title"].str.replace(PLACE_PREFIX, "", n=1, regex=True))

    # Filtre « zone lyonnaise » (règles explicites, voir lyon_filter.py)
    flags = [lyon_flag(t, p, loc) for t, p, loc in zip(df["title"], df["place_prefix"], df["locations"])]
    df["is_lyon"] = [f is not None for f in flags]
    df["lyon_rule"] = flags

    # Périmètre d'analyse : médias basés à Lyon en entier ; Le Progrès (régional : Ain, Jura, Loire…) filtré sur la zone lyonnaise
    df["in_scope"] = df["media_name"].isin(LYON_BASED_MEDIA) | ((df["media_name"] == "Le Progrès") & df["is_lyon"])

    # Type de contenu : l'agenda (fiches en gabarit) et le contenu sponsorisé ne sont pas de l'information éditoriale
    df["content_type"] = "editorial"
    df.loc[df["section"] == "pour-sortir", "content_type"] = "agenda"
    df.loc[df["section"].isin(["paroles-de-partenaires", "comparateur-telecom"]), "content_type"] = "sponsored"

    df["dup_title"] = df.duplicated(["media_name", "title"], keep=False)   # signalé, jamais supprimé
    df["tone_avg"] = pd.to_numeric(df["tone"].str.split(",").str[0], errors="coerce")

    corpus_flags = flag_low_coverage_days(df)
    df["low_coverage_day"] = corpus_flags
    df["in_analysis"] = df["in_scope"] & (df["content_type"] == "editorial")

    cols = ["article_id", "title", "title_clean", "url", "domain", "media_name", "seendate", "date", "date_source",
            "year", "month", "week", "week_start", "day_of_week", "language", "section", "place_prefix",
            "is_lyon", "lyon_rule", "in_scope", "content_type", "in_analysis", "low_coverage_day", "dup_title", "tone_avg", "sharing_image", "themes", "persons",
            "organizations", "locations"]
    corpus = df[cols].rename(columns={"sharing_image": "socialimage"}).sort_values(["date", "media_name"]).reset_index(drop=True)
    return corpus, stats


def coverage(corpus: pd.DataFrame, n_days: int) -> pd.DataFrame:
    g = corpus.groupby("media_name")
    t = pd.DataFrame({
        "Articles": g.size(),
        "Part du corpus (%)": (100 * g.size() / len(corpus)).round(1),
        "Articles / jour": (g.size() / n_days).round(1),
        "Jours couverts": g["date"].nunique(),
        "Première date": g["date"].min().dt.date, "Dernière date": g["date"].max().dt.date,
        "Titres présents (%)": (100 * g["title"].apply(lambda s: (s.str.len() > 0).mean())).round(1),
        "Titres en double": g["dup_title"].sum(),
        "Zone lyonnaise (%)": (100 * g["is_lyon"].mean()).round(1),
        "Articles zone lyonnaise": g["is_lyon"].sum(),
        "Articles dans le périmètre": g["in_scope"].sum(),
    })
    return t.sort_values("Articles", ascending=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True); TABLES.mkdir(parents=True, exist_ok=True)
    corpus, stats = build()
    corpus.to_parquet(OUT / "corpus.parquet", index=False)
    n_days = (corpus["seendate"].dt.normalize().max() - corpus["seendate"].dt.normalize().min()).days + 1
    cov = coverage(corpus, n_days)
    cov.to_csv(TABLES / "step2_coverage.csv")
    print(stats, f"| corpus : {len(corpus)} articles sur {n_days} jours")
    print(cov.to_string())


if __name__ == "__main__":
    main()
