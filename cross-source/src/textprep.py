"""Mots vides communs aux deux sources (liste NLTK française + liste du notebook Bluesky + mots du corpus presse)."""
import re
import unicodedata

import nltk
from nltk.corpus import stopwords

try:
    _NLTK = stopwords.words("french")
except LookupError:
    nltk.download("stopwords", quiet=True)
    _NLTK = stopwords.words("french")


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


CUSTOM = {"lyon", "lyonnais", "lyonnaise", "lyonnaises", "france", "rhône", "rhone", "octobre", "septembre", "août", "aout",
          "novembre", "juillet", "2026", "2025", "www", "com", "fr", "http", "https", "html", "url", "amp", "bluesky",
          "bsky", "social", "via", "tout", "tous", "toute", "toutes", "cette", "fait", "sont", "ont", "est", "été", "être",
          "plus", "aussi", "comme", "très", "bien", "peut", "faire", "alors", "encore", "juste", "quand", "veut", "faut",
          "voir", "ça", "cest", "jai", "nest", "dun", "dune", "mdr", "lol", "ptdr", "deux", "trois", "après", "avant",
          "contre", "près", "selon", "entre", "sous", "dont", "quoi", "comment", "pourquoi", "ici", "là", "chez", "aujourd",
          "hui", "demain", "hier", "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche", "ans", "an",
          "nouveau", "nouvelle", "grand", "grande", "premier", "première", "dernier", "dernière", "depuis", "cet", "ces"}
STOP_ACC = sorted(set(_NLTK) | CUSTOM | {strip_accents(w) for w in set(_NLTK) | CUSTOM})
TOKEN_PATTERN = r"(?u)\b[^\W\d_]{3,}\b"


def norm(text):
    """Texte normalisé pour les comptages de termes (minuscules, sans accents, sans URL ni mentions)."""
    t = strip_accents(str(text).lower())
    t = re.sub(r"http\S+|www\S+|@[\w\.\-]+", " ", t)
    t = re.sub(r"['’\-–—]", " ", t)
    return re.sub(r"[^\w\s]", " ", t)
