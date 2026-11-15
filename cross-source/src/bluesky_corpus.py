"""Construction du corpus Bluesky : règles reprises à l'identique de bluesky_topics.ipynb (Aaron Berton)."""
import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "bluesky"
LYON_REGEX = re.compile(
    r"lyon|villeurbann|vénissieux|venissieux|vaulx|bron\b|caluire|rhône|rhone|part-dieu|bellecour|"
    r"croix-rousse|confluence|perrache|guillotière|guillotiere|tcl\b|oullins|décines|decines|"
    r"saint-fons|givors|tassin|écully|ecully|meyzieu", re.I)
# Correctif par rapport au notebook source : « rhône » et « lyon » apparaissent aussi dans des noms qui ne désignent
# pas l'agglomération (Bouches-du-Rhône, vallée du Rhône, Gare de Lyon à Paris…). Même logique que le filtre presse.
NOT_LYON = re.compile(r"bouches[- ]du[- ]rh[ôo]ne|vall[ée]e du rh[ôo]ne|c[ôo]tes?[- ]du[- ]rh[ôo]ne|gare de lyon|golfe du lion", re.I)
EXCLUDE_AUTHORS = {"lyonpresquile.bsky.social", "lepolitichien.bsky.social", "actusnonstop.bsky.social"}
MIN_WORDS, MAX_PER_AUTHOR_DAY = 5, 5


def prep_text(t):
    """Texte pour les embeddings : la phrase est gardée, sans URL, mentions ni '#'."""
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"@[\w\.\-]+", " ", t)
    return re.sub(r"\s+", " ", t.replace("#", " ")).strip()


def load_raw():
    raw = pd.concat([pd.read_parquet(f) for f in sorted(RAW.glob("*.parquet"))], ignore_index=True)
    raw = raw.drop_duplicates("uri")
    raw["created_at"] = pd.to_datetime(raw["created_at"], utc=True, format="ISO8601", errors="coerce")
    raw = raw.dropna(subset=["created_at"])
    raw["date"] = raw["created_at"].dt.tz_convert("Europe/Paris").dt.tz_localize(None).dt.normalize()
    return raw


def build(raw):
    """Renvoie (corpus, entonnoir)."""
    funnel = {"collected": len(raw)}
    d = raw[(raw["date"] >= "2026-07-03") & (raw["date"] <= "2026-11-01")]
    d = d[d["langs"].apply(lambda l: l is not None and "fr" in list(l))]
    funnel["declared French"] = len(d)
    d = d.assign(is_lyon=d["text"].str.contains(LYON_REGEX), has_url=d["text"].str.contains(r"https?://|www\.", regex=True))
    stats = d.groupby("author").agg(posts=("uri", "size"), url_share=("has_url", "mean"),
                                    dup_share=("text", lambda s: 1 - s.nunique() / len(s)))
    suspects = set(stats[(stats.posts >= 50) & ((stats.url_share >= 0.5) | (stats.dup_share >= 0.3))].index)
    d = d[d["is_lyon"]]
    funnel["mention the Lyon area (source filter)"] = len(d)
    d = d[d["text"].map(lambda t: bool(LYON_REGEX.search(NOT_LYON.sub(" ", t))))]
    funnel["refined Lyon filter"] = len(d)
    d = d[~d["author"].isin(EXCLUDE_AUTHORS | suspects) & ~d["is_reply"]].copy()
    funnel["without bots, aggregators, replies"] = len(d)
    d["text_emb"] = d["text"].apply(prep_text)
    d = d[d["text_emb"].str.split().str.len() >= MIN_WORDS]
    d["_key"] = d["text_emb"].str.lower().str.replace(r"[^\w]+", " ", regex=True).str.strip()
    d = d.sort_values(["likes", "reposts"], ascending=False).drop_duplicates("_key")
    d["_rank"] = d.groupby(["author", "date"]).cumcount()
    d = d[d["_rank"] < MAX_PER_AUTHOR_DAY].drop(columns=["_key", "_rank"])
    d = d.sort_values("created_at").reset_index(drop=True)
    funnel["final corpus"] = len(d)
    return d, funnel, sorted(suspects)


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
