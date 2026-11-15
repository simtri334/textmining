# Cross-source — presse lyonnaise × Bluesky

Analyse croisée des deux volets du projet, utilisée dans le rapport final (`../report/`).

## Reproduire

Python 3.12 (PyTorch n'est pas encore disponible pour 3.14). Depuis `cross-source/` :

```bash
uv venv -p 3.12 .venv && uv pip install -p .venv/bin/python -r requirements.txt
.venv/bin/python scripts/01_collect_bluesky.py      # ~2 min, reprenable : un parquet par jour dans data/raw/bluesky/
.venv/bin/python scripts/02_build_and_embed.py      # corpus commun + embeddings MiniLM multilingue
.venv/bin/python scripts/03_sentiment.py            # sentiment XLM-T (même instrument pour les deux sources)
.venv/bin/python scripts/04_joint_topics.py         # BERTopic commun presse + Bluesky, stabilité sur 3 graines
.venv/bin/python scripts/05_cross_analyses.py       # composition, écho, synchronie, émergence, cadrages, liens
.venv/bin/python scripts/06_figures.py              # figures du rapport
../lyon-news-mining/.venv/bin/python scripts/07_press_figures.py   # figures « presse seule » (depuis textmining/)
```

Le corpus presse est lu dans `../lyon-news-mining/data/processed/` (à produire avant avec `build_corpus`).

## Choix importants (détaillés dans le rapport)

- **Collecte Bluesky** : API de recherche publique, mêmes 12 mots-clés et même période que `bluesky_topics.ipynb`. Sans authentification, la pagination ne fonctionne pas : une fenêtre de temps qui renvoie une page pleine est coupée en deux, récursivement.
- **Corpus Bluesky** : règles du notebook source (comptes automatiques, réponses, ≤ 5 posts/compte/jour, dédoublonnage), plus un **filtre territorial corrigé** (exclut Bouches-du-Rhône, vallée du Rhône, Gare de Lyon…).
- **Seuil d'écho** : calibré par l'annotation manuelle de 60 paires (`data/processed/echo_pairs_annotated.csv`).

## Données

`data/` est ignoré par Git : il contient des textes de posts et des identifiants de comptes. Les sorties de `outputs/` sont des agrégats.
