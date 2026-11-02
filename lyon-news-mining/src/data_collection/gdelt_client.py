"""Petit client pour l'API GDELT DOC 2.0 (https://api.gdeltproject.org/api/v2/doc/doc).

Contraintes de l'API (doc officielle + message renvoyé par le serveur) :
- 1 requête toutes les 5 secondes maximum ;
- fenêtre glissante d'environ 3 mois (paramètre `timespan` ou `startdatetime`/`enddatetime`) ;
- `mode=artlist` renvoie au plus 250 articles par requête (`maxrecords`).
"""
import time

import requests

BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
MIN_DELAY_S = 12.0         # la limite annoncée est de 5 s, mais des 429 sont observés dès 5,5 s : marge large
_last_call = 0.0


def query(params: dict, retries: int = 5) -> dict:
    """Appelle l'API avec limitation de débit et nouvelles tentatives. Renvoie le JSON décodé.

    Une réponse vide ({}), ou un message texte du serveur, est signalée par une exception
    explicite plutôt que silencieusement ignorée.
    """
    global _last_call
    params = {**params, "format": "json"}
    last_error = None
    for attempt in range(1, retries + 1):
        wait = MIN_DELAY_S - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.time()
        try:
            r = requests.get(BASE_URL, params=params, timeout=60,
                             headers={"User-Agent": "INSA-textmining-student-project"})
            if r.status_code == 200:
                if not r.text.strip():
                    return {}            # GDELT renvoie un corps vide quand il n'y a aucun résultat
                try:
                    return r.json()
                except ValueError:
                    last_error = f"réponse non JSON : {r.text[:150]!r}"
            else:
                last_error = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            last_error = repr(e)
        time.sleep(30 * attempt)          # backoff long : après un 429, GDELT bloque ~1-2 min
    raise RuntimeError(f"Échec de la requête {params} : {last_error}")


def artlist(search: str, timespan: str = "3months", maxrecords: int = 250, **extra) -> list[dict]:
    data = query({"query": search, "mode": "artlist", "maxrecords": maxrecords,
                  "timespan": timespan, "sort": "datedesc", **extra})
    return data.get("articles", [])


def timeline_volume(search: str, timespan: str = "3months") -> list[dict]:
    """Nombre d'articles réel par intervalle (non plafonné à 250), via timelinevolraw."""
    data = query({"query": search, "mode": "timelinevolraw", "timespan": timespan})
    timeline = data.get("timeline", [])
    return timeline[0]["data"] if timeline else []
