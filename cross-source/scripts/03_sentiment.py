"""Sentiment (négatif / neutre / positif) avec un même instrument pour les deux sources.

Modèle : cardiffnlp/twitter-xlm-roberta-base-sentiment (XLM-R entraîné sur des tweets en 8 langues dont le français).
Sortie : data/processed/sentiment.parquet (doc_id, p_neg, p_neu, p_pos, label, polarity = p_pos - p_neg)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

docs = pd.read_parquet(OUT / "docs.parquet", columns=["doc_id", "text"])
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForSequenceClassification.from_pretrained(NAME).eval()
device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
probs = []
with torch.no_grad():
    for i in range(0, len(docs), 64):
        batch = tok(docs["text"].iloc[i:i + 64].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        probs.append(torch.softmax(model(**batch).logits, dim=-1).cpu().numpy())
P = np.vstack(probs)                                   # ordre des classes du modèle : negative, neutral, positive
out = pd.DataFrame({"doc_id": docs["doc_id"], "p_neg": P[:, 0], "p_neu": P[:, 1], "p_pos": P[:, 2]})
out["label"] = np.array(["negative", "neutral", "positive"])[P.argmax(1)]
out["polarity"] = out["p_pos"] - out["p_neg"]
out.to_parquet(OUT / "sentiment.parquet", index=False)
print(out.label.value_counts().to_dict(), device)
