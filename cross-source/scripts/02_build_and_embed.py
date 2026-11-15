"""Construit les deux corpus (presse, Bluesky) dans un format commun et les encode avec le même modèle.

Sorties : data/processed/docs.parquet (une ligne = un document, source = press | bluesky)
          data/processed/embeddings.npy (même ordre)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.bluesky_corpus import build, load_raw  # noqa: E402

PRESS = ROOT.parent / "lyon-news-mining" / "data" / "processed"
OUT = ROOT / "data" / "processed"
EMB_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"      # même modèle que le notebook BERTopic Bluesky

press = pd.read_parquet(PRESS / "corpus.parquet")
press = press[press["in_analysis"]]
press_docs = pd.DataFrame({
    "doc_id": "P_" + press["article_id"], "source": "press", "outlet": press["media_name"],
    "author": press["media_name"], "date": press["date"], "timestamp": press["seendate"],
    "text": press["title_clean"].str.strip(), "url": press["url"], "likes": np.nan,
    "gdelt_tone": press["tone_avg"], "links": [[] for _ in range(len(press))]})

raw = load_raw()
bsky, funnel, suspects = build(raw)
bsky_docs = pd.DataFrame({
    "doc_id": "B_" + bsky["uri"].str.split("/").str[-1] + "_" + bsky["author_did"].str[-6:], "source": "bluesky",
    "outlet": "Bluesky", "author": bsky["author"], "date": bsky["date"],
    "timestamp": bsky["created_at"].dt.tz_convert("UTC").dt.tz_localize(None), "text": bsky["text_emb"], "url": bsky["uri"],
    "likes": bsky["likes"], "gdelt_tone": np.nan, "links": bsky["links"]})

docs = pd.concat([press_docs, bsky_docs], ignore_index=True)
docs["date"] = pd.to_datetime(docs["date"])
OUT.mkdir(parents=True, exist_ok=True)
docs.to_parquet(OUT / "docs.parquet", index=False)
json.dump({"bluesky_funnel": funnel, "bluesky_suspects_excluded": suspects,
           "press_docs": int(len(press_docs)), "bluesky_docs": int(len(bsky_docs))},
          open(ROOT / "outputs" / "tables" / "corpus_summary.json", "w"), indent=2)
# Liens partagés : sur tous les posts francophones du périmètre lyonnais (avant exclusion des comptes),
# car les comptes d'agrégation sont justement ceux qui relaient la presse
from src.bluesky_corpus import LYON_REGEX, NOT_LYON  # noqa: E402
raw_lyon = raw[raw["text"].map(lambda t: bool(LYON_REGEX.search(NOT_LYON.sub(" ", t))))
               & raw["langs"].apply(lambda l: l is not None and "fr" in list(l))]
raw_lyon[["uri", "author", "created_at", "date", "links", "is_reply", "likes"]].to_parquet(OUT / "bluesky_lyon_links.parquet", index=False)

from sentence_transformers import SentenceTransformer  # noqa: E402
model = SentenceTransformer(EMB_MODEL)
emb = model.encode(docs["text"].tolist(), batch_size=64, show_progress_bar=False, normalize_embeddings=True)
np.save(OUT / "embeddings.npy", emb.astype(np.float32))
print(funnel, docs.groupby("source").size().to_dict(), emb.shape)
