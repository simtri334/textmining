"""Nettoyage et normalisation des titres (étape 4).

Chaîne de traitement, du plus brut au plus normalisé :
    title_clean --(minuscules, élisions, ponctuation, chiffres)--> tokens_raw
                --(stopwords français + stopwords du corpus)-------> tokens_nostop
                --(lemmatisation simplemma + re-filtrage)----------> tokens_lemma
Chaque niveau est conservé pour pouvoir comparer avant/après dans le notebook 04.
"""
import re
from functools import lru_cache

import nltk
import pandas as pd
import simplemma
from nltk.corpus import stopwords

# Élisions françaises : « l'ancien » → « ancien », « qu'un » → « un »
ELISION = re.compile(r"\b(?:[ldjmtsnc]|qu|jusqu|lorsqu|puisqu|quoiqu)['’]", re.I)
TOKEN = re.compile(r"[a-zàâäçéèêëîïôöùûüÿœæ]+(?:-[a-zàâäçéèêëîïôöùûüÿœæ]+)*", re.I)

# Mots vides propres à ce corpus : présents dans presque tous les titres sans distinguer les sujets.
# « lyon », « rhône » : tout le corpus est lyonnais, ils dominent les fréquences mais n'apportent aucune information discriminante.
CORPUS_STOPWORDS = {"lyon", "lyonnais", "lyonnaise", "lyonnaises", "rhône", "rhone", "rhodanien", "rhodanienne",
                    "voici", "voilà", "découvrez", "découvrir", "alors", "comment", "pourquoi", "quoi", "plus", "moins",
                    "très", "tout", "tous", "toute", "toutes", "après", "avant", "encore", "déjà", "aussi", "ici",
                    "cette", "cet", "ces", "ans", "an", "jours", "jour", "ça", "c'est", "il", "elle", "ils", "elles"}
# Mots de calendrier et nombres : sans contenu thématique, et surtout corrélés au temps (« octobre » apparaîtrait
# faussement comme « émergent » en octobre). Adjectifs/adverbes génériques très fréquents dans les titres.
CALENDAR_STOPWORDS = {"janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre",
                      "novembre", "décembre", "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche",
                      "aujourd'hui", "hier", "demain", "week-end", "semaine", "mois", "année", "ce", "soir", "matin"}
GENERIC_STOPWORDS = {"deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf", "dix", "près", "contre", "nouveau",
                     "nouvelle", "nouveaux", "nouvelles", "grand", "grande", "grands", "grandes", "premier", "première",
                     "premiers", "premières", "dernier", "dernière", "derniers", "dernières", "mieux", "autre", "autres",
                     "même", "bien", "selon", "entre", "sous", "vers", "dont", "peut", "fait", "faits", "veut", "quand", "plein", "pleine"}
# Verbes « légers » conservés avant lemmatisation mais sans contenu thématique une fois lemmatisés
LEMMA_STOPWORDS = {"être", "avoir", "faire", "pouvoir", "devoir", "aller", "voir", "dire", "mettre", "prendre",
                   "venir", "passer", "rester", "falloir", "vouloir", "savoir", "donner", "trouver", "partir", "rendre"}


@lru_cache(maxsize=1)
def french_stopwords() -> frozenset:
    try:
        stopwords.words("french")
    except LookupError:                       # première exécution : télécharge la liste NLTK (quelques Ko)
        nltk.download("stopwords", quiet=True)
    return frozenset(stopwords.words("french")) | CORPUS_STOPWORDS | CALENDAR_STOPWORDS | GENERIC_STOPWORDS


# simplemma ramène « travaux » à « travail » (autre sens) : on garde la forme pour ces mots
LEMMA_EXCEPTIONS = {"travaux": "travaux", "élus": "élu", "bus": "bus", "auvergne-rhône-alpes": "auvergne-rhône-alpes", "alpes": "alpes"}


@lru_cache(maxsize=None)
def lemma(token: str) -> str:
    return LEMMA_EXCEPTIONS.get(token) or simplemma.lemmatize(token, lang="fr")


def tokenize(text: str) -> list[str]:
    """Minuscules, élisions retirées, mots alphabétiques uniquement (les chiffres sont supprimés)."""
    text = ELISION.sub("", text.lower().replace("’", "'"))
    return [t for t in TOKEN.findall(text) if len(t) >= 2]


def remove_stopwords(tokens: list[str]) -> list[str]:
    sw = french_stopwords()
    return [t for t in tokens if t not in sw and len(t) >= 3]


def lemmatize(tokens: list[str]) -> list[str]:
    sw = french_stopwords()
    out = (lemma(t) for t in tokens)
    return [t for t in out if t not in sw and t not in LEMMA_STOPWORDS and len(t) >= 3]


def add_token_columns(df: pd.DataFrame, text_col: str = "title_clean") -> pd.DataFrame:
    df = df.copy()
    df["tokens_raw"] = df[text_col].map(tokenize)
    df["tokens_nostop"] = df["tokens_raw"].map(remove_stopwords)
    df["tokens_lemma"] = df["tokens_nostop"].map(lemmatize)     # lemmatise les tokens déjà filtrés puis re-filtre
    return df
