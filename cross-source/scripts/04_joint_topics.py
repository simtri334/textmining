"""Modèle de thèmes commun presse + Bluesky (BERTopic, paramètres du notebook Bluesky), sur la fenêtre d'étude.

Sorties : data/processed/joint_topics.parquet, outputs/tables/joint_topics.csv, outputs/tables/joint_topics_stability.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import adjusted_rand_score
from umap import UMAP

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.textprep import STOP_ACC, TOKEN_PATTERN  # noqa: E402

P, T = ROOT / "data" / "processed", ROOT / "outputs" / "tables"
docs = pd.read_parquet(P / "docs.parquet")
emb = np.load(P / "embeddings.npy")
win = ((docs["date"] >= "2026-08-03") & (docs["date"] <= "2026-11-01")).values
docs, emb = docs[win].reset_index(drop=True), emb[win]
texts = docs["text"].tolist()


def fit(seed, target=40):
    umap = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=seed)
    hdb = HDBSCAN(min_cluster_size=max(15, int(len(docs) * 0.002)), min_samples=5, metric="euclidean",
                  cluster_selection_method="eom", prediction_data=True)
    vec = CountVectorizer(stop_words=STOP_ACC, ngram_range=(1, 2), min_df=5, token_pattern=TOKEN_PATTERN)
    m = BERTopic(umap_model=umap, hdbscan_model=hdb, vectorizer_model=vec,
                 representation_model=MaximalMarginalRelevance(diversity=0.4), calculate_probabilities=False,
                 language="multilingual")   # "english" (défaut) supprimerait les lettres accentuées des mots-clés
    topics, _ = m.fit_transform(texts, embeddings=emb)
    raw_out = float(np.mean(np.array(topics) == -1))
    n_raw = len(set(topics)) - (1 if -1 in topics else 0)
    if n_raw > target:
        m.reduce_topics(texts, nr_topics=target)
        topics = m.topics_
    topics_red = list(topics)
    new = m.reduce_outliers(texts, topics, strategy="embeddings", embeddings=emb, threshold=0.35)
    m.update_topics(texts, topics=new, vectorizer_model=vec, representation_model=MaximalMarginalRelevance(diversity=0.4))
    return m, np.array(new), np.array(topics_red), raw_out, n_raw


model, topics, topics_before, raw_out, n_raw = fit(42)
docs["topic"], docs["topic_before_reassign"] = topics, topics_before
docs[["doc_id", "source", "topic", "topic_before_reassign"]].to_parquet(P / "joint_topics.parquet", index=False)

n_src = docs["source"].value_counts()
rows = []
for t in sorted(set(topics) - {-1}):
    sub = docs[docs.topic == t]
    n_p, n_b = int((sub.source == "press").sum()), int((sub.source == "bluesky").sum())
    share_p, share_b = n_p / n_src["press"], n_b / n_src["bluesky"]
    rows.append({"topic": t, "docs": len(sub), "press": n_p, "bluesky": n_b, "share_press_%": 100 * share_p,
                 "share_bluesky_%": 100 * share_b, "log2_bsky_vs_press": np.log2((share_b + 1e-4) / (share_p + 1e-4)),
                 "keywords": ", ".join(w for w, _ in model.get_topic(t)[:10])})
tab = pd.DataFrame(rows).sort_values("docs", ascending=False)
tab.round(3).to_csv(T / "joint_topics.csv", index=False)

# stabilité : mêmes données, autres graines UMAP
stab = []
for s in (1, 2):
    _, t2, _, _, n2 = fit(s)
    stab.append({"seed": s, "ARI_vs_seed42": adjusted_rand_score(topics, t2), "raw_topics": n2})
pd.DataFrame(stab).to_csv(T / "joint_topics_stability.csv", index=False)
print(f"docs {len(docs)} | raw topics {n_raw}, raw outliers {raw_out:.0%} | final outliers {(topics == -1).mean():.1%}")
print(pd.DataFrame(stab))
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 110)
print(tab.round(2).to_string(index=False))
