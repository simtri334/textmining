# Bluesky — collecte, prétraitement et analyse des sujets

La partie Bluesky collecte les publications déclarées en français, les stocke dans SQLite, puis nettoie et dédoublonne les textes. Une analyse adaptée de [bluesky_topics.ipynb](../bluesky_topics.ipynb) ajoute la préparation du corpus, TF-IDF/K-Means, BERTopic, l'évolution temporelle et l'émergence lorsque l'historique le permet. Le code est organisé en notebooks.

## Installation

Python 3.11 ou plus récent. Depuis `textmining/bluesky-news-mining/` :

```powershell
python -m pip install -r requirements.txt
```

Ouvrir les notebooks dans VS Code et sélectionner le noyau Python où les dépendances sont installées : `websockets` pour la collecte, `pandas` pour le traitement et `ipykernel` pour exécuter les cellules.

Les dépendances de la référence TF-IDF/K-Means et des figures sont incluses dans `requirements.txt`. Pour BERTopic et la comparaison sémantique avec les médias, installer également :

```powershell
python -m pip install -r requirements-bertopic.txt
```

Cela ajoute `sentence-transformers`, `bertopic`, `umap-learn`, `hdbscan` et `pyarrow`. Le premier encodage peut télécharger MiniLM ; un modèle déjà en cache est réutilisé. Un environnement `.venv/` local, ignoré par Git, a été utilisé pour les vérifications : sélectionner son interpréteur dans VS Code si les dépendances y sont installées.

Le nettoyage lexical utilise **la liste française NLTK et la liste personnalisée exactes du notebook source**. Installer les mots vides dans le dossier local ignoré par Git :

```powershell
python -m nltk.downloader -d data/nltk_data stopwords
```

Le chargement des fonctions n'effectue aucun téléchargement implicite. Une liste absente provoque une erreur explicite indiquant cette commande.

## Organisation

```text
bluesky-news-mining/
├── notebooks/
│   ├── 00_collecteur_jetstream.ipynb
│   ├── 00_fonctions_base.ipynb
│   ├── 00_fonctions_analyse.ipynb
│   ├── 01_collecte.ipynb
│   ├── 02_base_et_controles.ipynb
│   ├── 03_nettoyage_et_dedoublonnage.ipynb
│   ├── 04_corpus_analyse_et_eda.ipynb
│   ├── 05_themes.ipynb
│   ├── 06_evolution_et_emergence.ipynb
│   └── 07_comparaison_medias.ipynb
├── posts.db
├── data/processed/           # corpus, affectations et caches locaux
├── outputs/tables/           # résultats locaux
├── outputs/figures/          # figures locales
├── tests/
│   ├── test_bluesky.py
│   └── test_analysis.py
├── requirements.txt
├── requirements-bertopic.txt
└── README.md
```

## Utilisation des notebooks

Les notebooks `00_*` définissent les fonctions communes sans lancer de collecte, charger la base ou entraîner les modèles. Les notebooks 01 à 07 les chargent automatiquement ; exécuter leurs cellules de haut en bas.

| Notebook | Fonctionnement implémenté |
|---|---|
| [00_collecteur_jetstream.ipynb](notebooks/00_collecteur_jetstream.ipynb) | Connexion Jetstream, détection des liens, création de la table, traitement des événements et reconnexion. |
| [00_fonctions_base.ipynb](notebooks/00_fonctions_base.ipynb) | Lecture SQLite, nettoyage d'un texte et prétraitement complet. |
| [00_fonctions_analyse.ipynb](notebooks/00_fonctions_analyse.ipynb) | Préparation, modèles thématiques, cache d'embeddings, émergence et comparaison aux médias. |
| [01_collecte.ipynb](notebooks/01_collecte.ipynb) | Paramètres et lancement de la collecte, puis comptage des publications stockées. |
| [02_base_et_controles.ipynb](notebooks/02_base_et_controles.ipynb) | Schéma SQLite, effectifs, dates, langues déclarées, unicité des URI et aperçu des textes bruts. |
| [03_nettoyage_et_dedoublonnage.ipynb](notebooks/03_nettoyage_et_dedoublonnage.ipynb) | Nettoyage détaillé, retrait des textes courts, dédoublonnage et bilan des effectifs. |
| [04_corpus_analyse_et_eda.ipynb](notebooks/04_corpus_analyse_et_eda.ipynb) | Filtres d'analyse, statistiques des auteurs, exploration et export du corpus. |
| [05_themes.ipynb](notebooks/05_themes.ipynb) | TF-IDF/K-Means, silhouette, stabilité ARI, BERTopic, mots-clés et exemples. |
| [06_evolution_et_emergence.ipynb](notebooks/06_evolution_et_emergence.ipynb) | Parts temporelles des thèmes, score d'émergence et regroupement d'événements par Jaccard. |
| [07_comparaison_medias.ipynb](notebooks/07_comparaison_medias.ipynb) | Comparaison sémantique et temporelle optionnelle avec la presse lyonnaise. |

### Collecte

Dans le notebook 01, définir les paramètres suivants puis exécuter la cellule contenant `await collect_posts(...)` :

```python
RUN_COLLECTION = True
MAX_POSTS = 1000
DURATION_MINUTES = None
```

La collecte est désactivée par défaut (`RUN_COLLECTION = False`). Elle conserve la base existante. `MAX_POSTS` compte les nouvelles URI enregistrées pendant l'exécution ; des suppressions reçues peuvent réduire le nombre final de lignes. `DURATION_MINUTES` permet de limiter la durée ; `None` ne fixe pas de durée maximale. Interrompre la cellule arrête la collecte.

Le collecteur écoute la collection `app.bsky.feed.post` et conserve les posts dont la liste des langues déclarées contient exactement `fr`. Les posts multilingues sont acceptés ; les posts sans langue déclarée sont ignorés. Ce filtre ne détecte pas automatiquement la langue réelle du texte. Les réponses et les publications sans lien sont incluses.

### Base SQLite

La table `posts` conserve `uri` (clé primaire), `did` (auteur), `created_at`, `text`, `langs`, `is_reply` et `has_link`. Les créations et modifications sont enregistrées par URI. Les suppressions reçues retirent les posts de la base ; une modification ne déclarant plus `fr` retire également l'ancienne ligne.

Le collecteur tente de se reconnecter après une interruption réseau avec un curseur en mémoire. Le curseur n'est pas sauvegardé après l'arrêt ; une nouvelle exécution repart du direct. La reprise dépend de la rétention du serveur et n'est pas garantie sans lacune.

### Nettoyage et dédoublonnage

Les notebooks 02 et 03 peuvent être exécutés directement sur la base existante, sans connexion réseau. Ils chargent toutes les publications et normalisent leurs dates en UTC.

Le notebook 03 applique successivement :

1. Vérification de la présence de `fr` dans les langues déclarées.
2. Retrait des URL commençant par `http://`, `https://` ou `www.`.
3. Normalisation des espaces, tabulations et retours à la ligne.
4. Retrait des textes nettoyés de moins de 3 caractères.
5. Dédoublonnage exact de `clean_text`, en conservant la première occurrence chargée (ordre de date décroissant, puis URI).

La casse, les accents, la ponctuation et les emojis sont conservés. Deux textes ne différant que par une URL peuvent devenir identiques après nettoyage.

Le résultat est disponible dans `posts_unique`. Les opérations s'effectuent en mémoire : elles ne modifient pas `posts.db` et ne créent pas de fichier exporté.

## Analyse intégrée depuis `bluesky_topics.ipynb`

Exécuter **04 → 05 → 06**, puis éventuellement 07. Le notebook 04 réutilise `load_posts` et `preprocess_posts` : il n'est pas nécessaire d'avoir exécuté le notebook 03 dans le même noyau. Les étapes suivantes lisent les exports locaux de la précédente. Le notebook source est conservé sans modification.

### Corpus d'analyse et EDA

Les paramètres du notebook 04 sont :

- `INCLUDE_REPLIES = False` : exclusion des réponses pour l'analyse uniquement ;
- `LYON_ONLY = False` : corpus francophone général ; activer pour reprendre le périmètre lyonnais de la source ;
- `MIN_WORDS = 5` : longueur minimale du texte destiné aux embeddings ;
- `MAX_POSTS_PER_AUTHOR_DAY = 5` : plafond par auteur et par jour ;
- `AUTO_EXCLUDE_PROLIFIC = False` : les fortes activités sont signalées, sans être automatiquement qualifiées de bots ou exclues ;
- `EXCLUDE_AUTHORS` : exclusions manuelles facultatives par DID.

La base Jetstream ne fournit pas les likes, reposts et handles du notebook source. Les auteurs sont identifiés par DID et les doublons normalisés sont départagés par récence, sans créer de faux compteurs d'engagement.

Deux textes sont conservés : `text_emb` garde les phrases sans URL, mentions ou marqueurs `#` ; `texte_clean` est calculé sur le texte original avec le nettoyage exact de la source : minuscules, accents retirés, URL et mentions retirées, apostrophes/tirets séparés, ponctuation retirée, mots vides NLTK et personnalisés supprimés, retrait des mots purement numériques ou de moins de 3 caractères. Les mots « Lyon » et « France » sont retirés comme dans la source, même en mode francophone général.

La compatibilité porte sur le traitement d'analyse à corpus fourni identique. L'entrée reste différente : collecte Jetstream, langue déclarée, prétraitement exact existant, territoire optionnel et pas de compteurs d'engagement. Les exclusions par handle et le choix du doublon le plus liké ne peuvent donc pas être reproduits sur cette base. Aucun engagement n'est inventé.

### Modèles thématiques

Le notebook 05 reprend **TF-IDF + K-Means avec les paramètres de la source** : `min_df=5`, `max_df=0.5`, unigrammes/bigrammes, TF sous-linéaire, 14 groupes, `n_init=10`, graine 42. La silhouette utilise la distance euclidienne par défaut, comme la source. La stabilité ARI compare ce modèle à trois entraînements avec `n_init=3` et graines 1, 2 et 3. Le nombre de groupes n'est plus ajusté automatiquement ; un corpus insuffisant provoque une erreur explicite.

Les vecteurs TF-IDF nuls restent dans l'entraînement et sont affectés à un groupe, comme dans la source. Leur nombre est indiqué dans les diagnostics pour rendre visible cette limite de la méthode. Cette référence diffère de la chaîne presse utilisant lemmatisation et LSA.

**BERTopic** est activé avec `RUN_BERTOPIC = True`. Mettre la variable à `False` pour n'exécuter que K-Means. Les paramètres sont ceux de la source : MiniLM multilingue, UMAP (`n_neighbors=15`, `n_components=5`, `min_dist=0`, métrique cosinus, graine 42 et initialisation par défaut), HDBSCAN (`max(15, 0.002 × nombre de posts)` pour la taille minimale, `min_samples=5`, sélection `eom`), c-TF-IDF avec mots vides accentués de la source et `min_df=5`, MMR de diversité 0,4. Le cache est validé par le modèle, les URI et les textes dans leur ordre. `TARGET_TOPICS = 30` borne une fusion éventuelle sans imposer 30 groupes.

Les outliers sont réaffectés par défaut comme dans la source : `REASSIGN_OUTLIERS = True`, stratégie embeddings et seuil 0,35. Les étiquettes brutes et finales sont conservées séparément. Les libellés sont des mots-clés automatiques, pas des résumés d'actualité validés. Ni silhouette ni ARI ne mesurent la précision journalistique.

### Évolution et émergence

Le notebook 06 exporte les parts quotidiennes et hebdomadaires des thèmes, avec pour dénominateur les posts affectés à un thème (outliers exclus), comme dans la source.

Il reprend le score de la source : `z = (observé − attendu) / sqrt(attendu + 1)`. Le taux attendu est estimé sur **28 jours de référence**, avant une fenêtre récente de **3 jours**. Les seuils sont `z ≥ 4`, au moins 6 posts et 2 auteurs. Pour chaque terme, le pic maximal respectant le seuil de posts est sélectionné avant vérification du nombre d'auteurs, comme dans la source.

Les termes sont regroupés sur les 600 premiers candidats par distance de Jaccard, liaison moyenne et seuil 0,8, sans séparation préalable par date de pic. S'il y a moins de deux termes, aucun événement n'est regroupé, comme dans la source. Les exemples sont classés par likes si cette colonne existe, sinon ils utilisent les premiers posts disponibles.

Le calcul conserve la règle source `max(nombre de posts de référence, 1)` et n'ajoute plus de filtre de couverture. Une référence sans collecte peut donc produire un pic artificiel : les résultats nécessitent une inspection des périodes observées. Le score est heuristique, pas une p-valeur ; il diffère du score hebdomadaire pondéré par média du projet presse. Le seul garde-fou structurel est de signaler un historique trop court plutôt que de provoquer une erreur sur un tableau vide.

**Le corpus actuel couvre une seule journée : l'émergence n'est pas calculable avec ces paramètres.** Le diagnostic `insufficient_history` explique cette abstention. Un tableau vide ne signifie pas qu'aucune actualité n'existe.

### Comparaison aux médias

Le notebook 07 est désactivé par défaut (`RUN_COMPARISON = False`). Il lit le vrai fichier `../lyon-news-mining/data/processed/article_topics.parquet` (colonne `km`) et récupère les titres manquants dans `corpus.parquet` par une jointure sur `article_id`. Les deux sources sont encodées avec le même modèle et leurs numéros de thèmes ne sont jamais assimilés.

Une comparaison nécessite des dates communes et un périmètre compatible. La collecte du 8 novembre ne chevauche pas le corpus presse, qui s'arrête au 1er novembre : si la comparaison est activée, le traitement signale `no_date_overlap`. Comme dans la source, les séries normalisées sont lissées sur 3 jours et les corrélations explorent des décalages de −7 à +7 jours. Les corrélations indéfinies (moins de deux observations ou variance nulle) restent manquantes. Elles n'établissent pas qu'une source publie une information en premier.

### Résultats de la première analyse intégrée

Après alignement du nettoyage sur la source, la collecte actuelle conserve **275 posts** pour l'analyse, à partir des 930 textes prétraités :

| Méthode | Résultat |
|---|---|
| TF-IDF/K-Means | 14 groupes sur les 275 posts ; 118 vecteurs nuls inclus comme dans la source ; silhouette euclidienne ≈ 0,455 ; stabilité ARI ≈ 0,579 |
| BERTopic | Paramètres source non applicables à ce corpus après mise à jour des topics : `min_df=5` élimine tout le vocabulaire ; aucun assouplissement automatique |
| Émergence | Non évaluée : historique d'une journée au lieu des 31 jours nécessaires |
| Comparaison presse | Désactivée par défaut ; périodes actuellement disjointes |

Les thèmes K-Means restent exploratoires. La présence de nombreux vecteurs nuls limite l'interprétation de la silhouette. BERTopic n'est pas déclaré réussi lorsque la méthode source échoue : le diagnostic `source_parameters_not_applicable` explique le problème. Les anciens chiffres BERTopic de la première adaptation ne s'appliquent plus à ce traitement aligné.

Les cas non calculables (vocabulaire vide, historique trop court ou dates disjointes) sont signalés au lieu de provoquer une erreur non expliquée. Cela ne change pas les paramètres ou les calculs des cas évaluables. Les différences restantes concernent l'entrée des données : la collecte Jetstream n'est pas la collecte historique lyonnaise et ne possède pas les compteurs d'engagement utilisés par la source.

### Exports locaux

Les étapes 04 à 07 écrivent dans `data/processed/`, `outputs/tables/` et `outputs/figures/`. Ces dossiers sont ignorés par Git car ils contiennent des textes, des identifiants d'auteurs et des caches. La base brute reste inchangée.

Les notebooks Bluesky sont versionnés sans leurs sorties d'exécution pour éviter d'y inclure les textes et identifiants des posts. Après une exécution locale, effacer les sorties avant de les ajouter à Git.

Les principaux fichiers sont `corpus_analyse.csv`, `post_topics.csv`, `topic_diagnostics.json`, les tableaux `05_*` et `06_*`, ainsi que les diagnostics d'émergence et de comparaison. Relancer les étapes suivantes après modification de la base ou des paramètres : ces exports sont des instantanés.

## Dernière collecte vérifiée

Collecte renouvelée le **8 novembre 2026** après vidage de la base :

| Mesure | Résultat |
|---|---:|
| Posts bruts déclarés `fr` | 1 000 |
| Auteurs distincts | 603 |
| Textes trop courts retirés après nettoyage | 52 |
| Doublons exacts retirés | 18 |
| **Textes nettoyés et uniques** | **930** |

Les 1 000 posts bruts restent dans SQLite. Les effectifs ci-dessus proviennent de l'exécution des notebooks sur cette collecte ; leurs sorties ne sont pas versionnées. Les effectifs évolueront si la base est alimentée à nouveau.

## Vérifications

Depuis le sous-projet :

```powershell
python -m unittest discover -s tests -v
```

Les **20 tests** couvrent la collecte, le prétraitement, les cas limites de l'analyse et des comparaisons directes avec les cellules du notebook source. À entrées identiques, ces tests vérifient l'égalité du nettoyage lexical, des affectations/métriques K-Means, des scores d'émergence et du regroupement Jaccard. Ils utilisent des données simulées et des bases temporaires, sans accès réseau, une fois les dépendances et mots vides NLTK installés.

Les tests initiaux vérifient le cycle création / modification / suppression, les langues déclarées, les liens, SQLite, les dates, le nettoyage et le dédoublonnage, y compris sur une base vide. Les tests synthétiques ne valident pas la pertinence journalistique : celle-ci nécessite une lecture et une annotation des groupes produits.
