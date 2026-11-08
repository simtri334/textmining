# Bluesky — collecte et prétraitement

La partie Bluesky collecte les publications déclarées en français, les stocke dans SQLite, nettoie les textes et supprime les doublons exacts en mémoire. Le code est organisé en notebooks.

## Installation

Python 3.11 ou plus récent. Depuis `textmining/bluesky-news-mining/` :

```powershell
python -m pip install -r requirements.txt
```

Ouvrir les notebooks dans VS Code et sélectionner le noyau Python où les dépendances sont installées : `websockets` pour la collecte, `pandas` pour le traitement et `ipykernel` pour exécuter les cellules.

## Organisation

```text
bluesky-news-mining/
├── notebooks/
│   ├── 00_collecteur_jetstream.ipynb
│   ├── 00_fonctions_base.ipynb
│   ├── 01_collecte.ipynb
│   ├── 02_base_et_controles.ipynb
│   └── 03_nettoyage_et_dedoublonnage.ipynb
├── posts.db
├── tests/test_bluesky.py
├── requirements.txt
└── README.md
```

## Utilisation des notebooks

Les deux notebooks `00_*` définissent les fonctions communes, sans lancer de collecte ni charger la base. Les notebooks 01, 02 et 03 les chargent automatiquement ; exécuter leurs cellules de haut en bas.

| Notebook | Fonctionnement implémenté |
|---|---|
| [00_collecteur_jetstream.ipynb](notebooks/00_collecteur_jetstream.ipynb) | Connexion Jetstream, détection des liens, création de la table, traitement des événements et reconnexion. |
| [00_fonctions_base.ipynb](notebooks/00_fonctions_base.ipynb) | Lecture SQLite, nettoyage d'un texte et prétraitement complet. |
| [01_collecte.ipynb](notebooks/01_collecte.ipynb) | Paramètres et lancement de la collecte, puis comptage des publications stockées. |
| [02_base_et_controles.ipynb](notebooks/02_base_et_controles.ipynb) | Schéma SQLite, effectifs, dates, langues déclarées, unicité des URI et aperçu des textes bruts. |
| [03_nettoyage_et_dedoublonnage.ipynb](notebooks/03_nettoyage_et_dedoublonnage.ipynb) | Nettoyage détaillé, retrait des textes courts, dédoublonnage et bilan des effectifs. |

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

## Dernière collecte vérifiée

Collecte renouvelée le **8 novembre 2026** après vidage de la base :

| Mesure | Résultat |
|---|---:|
| Posts bruts déclarés `fr` | 1 000 |
| Auteurs distincts | 603 |
| Textes trop courts retirés après nettoyage | 52 |
| Doublons exacts retirés | 18 |
| **Textes nettoyés et uniques** | **930** |

Les 1 000 posts bruts restent dans SQLite. Les sorties des notebooks ont été actualisées sur cette collecte. Les effectifs évolueront si la base est alimentée à nouveau.

## Vérifications

Depuis le sous-projet :

```powershell
python -m unittest discover -s tests -v
```

Les six tests vérifient le cycle création / modification / suppression, le changement de langue déclarée, la détection des liens, la lecture SQLite, les dates, le nettoyage et le dédoublonnage, y compris sur une base vide. Ils chargent les fonctions des notebooks et utilisent des données simulées et des bases temporaires, sans accès réseau.

Lors de la dernière vérification, les six tests sont passés et les cinq notebooks ont été exécutés avec la collecte désactivée. Les contrôles sur les données réelles ont validé l'intégrité SQLite, les langues déclarées, l'unicité des URI et la concordance entre le nettoyage détaillé et la fonction commune `preprocess_posts`.
