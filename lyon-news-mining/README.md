# Lyon News Mining — détection de sujets émergents dans la presse lyonnaise (GDELT)

Projet Text Mining / NLP, INSA Lyon (5IF). Question : comment les sujets traités par les médias locaux lyonnais évoluent-ils dans le temps,
et peut-on détecter automatiquement les thématiques et les sujets émergents ?

## État d'avancement
- [x] **Collecte et corpus** : `notebooks/02_corpus_et_couverture.ipynb`
- [x] **Étape 3, EDA** : `03_eda.ipynb`
- [x] **Étape 4, analyse textuelle** : `04_analyse_textuelle.ipynb`
- [x] **Étape 5, thématiques (TF-IDF, K-Means, LDA)** : `05_themes.ipynb`
- [x] **Étape 6, sujets émergents** : `06_emergence.ipynb`
- [x] **Étape 7, figures et interprétation** : `07_visualisations.ipynb`
- [ ] Rapport et soutenance : plan dans `docs/plan_rapport.md`

Ordre d'exécution : 02 → 03 → 04 → 05 → 06 → 07 (05 produit `data/processed/article_topics.parquet`, utilisé par 06 et 07).

## Source des données (point important)
L'API GDELT DOC 2.0 (`mode=artlist`) était inutilisable lors de nos tests (HTTP 429 puis réponses vides, y compris pour un média témoin), nous n'avons donc **aucun résultat issu de l'API**
(le client reste dans `src/data_collection/gdelt_client.py`). Le corpus provient des **fichiers GDELT GKG 2.0 « translation »** (flux des médias non anglophones, toutes les 15 min),
téléchargés en mémoire et filtrés sur les domaines lyonnais ; seules les lignes retenues sont stockées.

## Reproduire
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.data_collection.gkg_collector 2026-08-03 2026-11-01   # ~30 s/jour, reprenable (jours déjà collectés ignorés)
python -m src.preprocessing.build_corpus                            # → data/processed/corpus.parquet
jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb   # ou exécuter les notebooks 02 → 07 dans l'ordre
```
Dans VS Code, ouvrir ce dossier et choisir le kernel `.venv` (ou « Python (lyon-news-mining) »).

## Structure
```
data/raw/gkg/          1 fichier .csv.gz par jour : lignes GKG des médias ciblés (brut, non modifié)
data/processed/        corpus.parquet (dédoublonné, dates et rubriques dérivées)
src/data_collection/   gkg_collector.py (collecte), gdelt_client.py (API DOC, non utilisable), test_domains.py (test API)
src/preprocessing/     build_corpus.py, lyon_filter.py (périmètre lyonnais), text_cleaning.py (étape 4)
src/modeling/          topics.py (TF-IDF, K-Means/LSA, LDA, cohérence NPMI)
src/analysis/          emergence.py (score d'émergence, regroupement, trajectoires)
docs/                  plan_rapport.md
notebooks/             analyses  ·  outputs/figures, outputs/tables : résultats pour le rapport
```

## Résultats principaux
13 998 articles uniques du 3 août au 1er novembre 2026 : Le Progrès 80 %, Lyon Capitale 13 %, LyonMag 6 %, BFM Lyon < 1 % ; Rue89Lyon, Tribune de Lyon et Actu Lyon absents de GDELT. **Périmètre d'analyse : 4 936 titres.**
- 14 thèmes K-Means (qualité modeste : silhouette faible, stabilité ARI ≈ 0,25) ; LDA peu adapté aux titres courts.
- 87 termes émergents (z ≥ 4, ≥ 6 titres, ≥ 2 médias) regroupés en 14 événements ; 31 des 32 premiers correspondent à un événement identifié à la lecture : mouvement lycéen (fin sept. – oct.), émeutes du 17 octobre et couvre-feu, affaire Quentin Deranque, attentat de Trappes…
- Les médias ont des profils thématiques distincts (V de Cramér 0,24).

## Limites de la source
- Titres seulement (pas de texte intégral) ; `sourcecountry` absent du GKG ; tous les articles sont en français.
- Une seule saison de 13 semaines : saisonnalité et tendances longues non mesurables.
- Le Progrès est un quotidien régional (Ain, Jura, Loire…) : filtré sur la zone lyonnaise par une heuristique (préfixe de lieu, mots du titre, géolocalisation GDELT).
- La date est celle de détection GDELT, corrigée par la date de l'URL quand elle existe (Le Progrès uniquement).
- GDELT n'indexe qu'une partie des articles et a un trou de couverture pour Le Progrès les 11–13 octobre 2026.
