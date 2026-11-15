"""Collecte Bluesky (API de recherche publique) : mêmes requêtes et même période que bluesky_topics.ipynb.

Un fichier parquet par jour (UTC) dans data/raw/bluesky/ : la collecte est reprenable.
En plus des champs du notebook source, on garde les liens partagés (carte de lien et liens du texte),
pour mesurer quels médias les posts citent.

Usage (depuis cross-source/) : python scripts/01_collect_bluesky.py
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "bluesky"
URL = "https://api.bsky.app/xrpc/app.bsky.feed.searchPosts"
START, END = pd.Timestamp("2026-07-03"), pd.Timestamp("2026-11-01")   # 31 jours de référence + fenêtre d'étude
QUERIES = ["lyon", "lyonnais", "lyonnaise", "métropole de lyon", "grand lyon", "villeurbanne",
           "rhône", "part-dieu", "bellecour", "croix-rousse", "vénissieux", "tcl"]
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "insa-textmining-student-project"


def get(params, retries=6):
    for attempt in range(retries):
        try:
            r = requests.get(URL, params=params, timeout=30, headers={"User-Agent": "insa-textmining-student-project"})
            if r.status_code == 200:
                try:
                    return r.json()
                except ValueError:          # réponse vide ou non JSON : on retente
                    time.sleep(min(30, 2 ** attempt))
                    continue
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(60, 2 ** attempt))
                continue
            r.raise_for_status()
        except requests.RequestException:
            time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"échec {params}")


def links(post):
    rec, out = post.get("record", {}), []
    ext = (rec.get("embed") or {}).get("external") or ((rec.get("embed") or {}).get("media") or {}).get("external")
    if ext and ext.get("uri"):
        out.append(ext["uri"])
    for facet in rec.get("facets") or []:
        for feat in facet.get("features", []):
            if feat.get("$type", "").endswith("#link") and feat.get("uri"):
                out.append(feat["uri"])
    return list(dict.fromkeys(out))


def parse(p, q):
    rec = p.get("record", {})
    return {"uri": p["uri"], "author": p["author"]["handle"], "author_did": p["author"]["did"],
            "text": rec.get("text", ""), "langs": rec.get("langs") or [], "created_at": rec.get("createdAt"),
            "likes": p.get("likeCount", 0), "reposts": p.get("repostCount", 0),
            "replies": p.get("replyCount", 0), "quotes": p.get("quoteCount", 0),
            "is_reply": "reply" in rec, "has_embed": "embed" in rec, "links": links(p), "query": q}


def fetch(q, since, until, depth=0):
    """Sans authentification, la pagination par curseur ne fonctionne pas (page 2 vide) : une fenêtre
    qui renvoie une page pleine (100 posts) est coupée en deux, récursivement, jusqu'à passer sous la limite."""
    data = get({"q": q, "lang": "fr", "limit": 100, "sort": "latest",
                "since": since.strftime("%Y-%m-%dT%H:%M:%S.000Z"), "until": until.strftime("%Y-%m-%dT%H:%M:%S.000Z")})
    posts = data.get("posts", [])
    if len(posts) >= 100 and (until - since) > pd.Timedelta(minutes=10) and depth < 12:
        mid = since + (until - since) / 2
        return fetch(q, since, mid, depth + 1) + fetch(q, mid, until, depth + 1)
    return [parse(p, q) for p in posts]


def collect_day(day):
    f = RAW / f"{day:%Y-%m-%d}.parquet"
    if f.exists():
        return day, -1
    rows = [r for q in QUERIES for r in fetch(q, day, day + pd.Timedelta(days=1))]
    df = pd.DataFrame(rows).drop_duplicates("uri") if rows else pd.DataFrame(columns=["uri"])
    df.to_parquet(f.with_suffix(".tmp"), index=False)
    f.with_suffix(".tmp").rename(f)
    return day, len(df)


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    days = list(pd.date_range(START, END, freq="D"))
    t0 = time.time()
    with ThreadPoolExecutor(6) as ex:
        for day, n in ex.map(collect_day, days):
            print(f"{day:%Y-%m-%d} : {n} posts ({time.time() - t0:,.0f}s)", flush=True)
