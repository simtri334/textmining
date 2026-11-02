"""Détection de sujets émergents (étape 6) : mots, expressions et thèmes, au pas hebdomadaire.

Principe : un sujet est « émergent » quand sa **part relative** dans la production d'une semaine dépasse nettement sa part passée
(les 4 semaines précédentes) ; un sujet seulement *fréquent* mais stable n'est pas émergent.

Deux précautions issues de l'EDA (notebook 03) :
  1. Composition par média : le poids de chaque média varie d'une semaine à l'autre (panne GDELT du Progrès du 12 octobre).
     Les parts sont donc calculées PAR MÉDIA puis recombinées avec des poids fixes (parts moyennes des médias sur la période).
  2. Seuls les médias à volume suffisant entrent dans les séries (Le Progrès, Lyon Capitale, LyonMag).
"""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from src.preprocessing.text_cleaning import french_stopwords

MAIN_MEDIA = ["Le Progrès", "Lyon Capitale", "LyonMag"]


def term_sets(tokens_raw, tokens_lemma):
    """Termes d'un titre : lemmes (unigrammes) + bigrammes de mots consécutifs (sur les formes brutes) commençant et finissant par un mot plein."""
    sw = french_stopwords()
    out = []
    for raw, lem in zip(tokens_raw, tokens_lemma):
        terms = set(lem)
        for a, b in zip(raw, raw[1:]):
            if a not in sw and b not in sw and len(a) > 2 and len(b) > 2:
                terms.add(f"{a} {b}")
        out.append(list(terms))
    return out


def binary_matrix(sets, min_df=10):
    vec = CountVectorizer(analyzer=lambda x: x, min_df=min_df, binary=True)
    B = vec.fit_transform(sets).tocsr()
    return B, np.array(vec.get_feature_names_out())


def weekly_counts(B, df, weeks, media=MAIN_MEDIA):
    """counts[m, w, t] = nombre de titres du média m, semaine w, contenant le terme t ; n[m, w] = nombre de titres."""
    counts = np.zeros((len(media), len(weeks), B.shape[1]))
    n = np.zeros((len(media), len(weeks)))
    wk = df["week_start"].values
    for i, m in enumerate(media):
        is_m = (df["media_name"] == m).values
        for j, w in enumerate(weeks):
            idx = np.where(is_m & (wk == w))[0]
            n[i, j] = len(idx)
            if len(idx):
                counts[i, j] = np.asarray(B[idx].sum(axis=0)).ravel()
    return counts, n


def media_weights(df, media=MAIN_MEDIA):
    s = df[df["media_name"].isin(media)]["media_name"].value_counts()
    return np.array([s[m] for m in media]) / s.sum()


def balanced_share(counts, n, pi):
    """Part hebdomadaire d'un terme = Σ_m π_m · (titres du média m avec le terme / titres du média m)."""
    p_m = counts / np.maximum(n[:, :, None], 1)
    return (pi[:, None, None] * p_m).sum(axis=0)             # [w, t]


def binom_var(p, n, pi):
    """Variance de la part recombinée sous H0 (binomiale par média) : Σ_m π_m² p(1-p)/n_m ; p : [t], n : [m] → [t]."""
    return (pi[:, None] ** 2 * (p[None, :] * (1 - p[None, :])) / np.maximum(n[:, None], 1)).sum(axis=0)


def emergence_scores(counts, n, pi, base_weeks=4, balanced=True):
    """Pour chaque semaine t ≥ base_weeks et chaque terme : z = (p_t − p0) / se, p0 = part moyenne des `base_weeks` semaines précédentes.

    Le plancher p0 ≥ 1/(2 N) évite qu'un terme absent du passé reçoive un score infini ; se² = variance binomiale de la semaine t
    + variance de la moyenne de base (deux échantillons indépendants). Renvoie (z, p, p0) de forme [w, t] (NaN avant base_weeks).
    """
    W, T = counts.shape[1], counts.shape[2]
    if balanced:
        p = balanced_share(counts, n, pi)
    else:                                                     # témoin : parts brutes, tous médias confondus
        p = counts.sum(axis=0) / np.maximum(n.sum(axis=0)[:, None], 1)
        pi_raw = None
    z, p0s = np.full((W, T), np.nan), np.full((W, T), np.nan)
    for t in range(base_weeks, W):
        base = slice(t - base_weeks, t)
        p0 = p[base].mean(axis=0)
        floor = 1.0 / (2 * n[:, t].sum())
        p0 = np.maximum(p0, floor)
        if balanced:
            var_t = binom_var(p0, n[:, t], pi)
            var_b = np.mean([binom_var(p0, n[:, b], pi) for b in range(t - base_weeks, t)], axis=0) / base_weeks
        else:
            var_t = p0 * (1 - p0) / max(n[:, t].sum(), 1)
            var_b = np.mean([p0 * (1 - p0) / max(n[:, b].sum(), 1) for b in range(t - base_weeks, t)], axis=0) / base_weeks
        z[t] = (p[t] - p0) / np.sqrt(var_t + var_b)
        p0s[t] = p0
    return z, p, p0s


def classify_trajectory(p_series, peak, p0_peak, after=2):
    """'montée durable' si, parmi les semaines suivant le pic, l'une garde au moins la moitié de l'excès ; 'pic ponctuel' sinon ;
    'indéterminé' si le pic est trop près de la fin de la période pour trancher."""
    if peak + after >= len(p_series):
        return "indéterminé (fin de période)"
    excess = p_series[peak] - p0_peak
    later = p_series[peak + 1: peak + 1 + after]
    return "montée durable" if (later >= p0_peak + 0.5 * excess).any() else "pic ponctuel"


def detect_events(counts, n, pi, terms, B, df, weeks, z_min=4.0, n_min=6, media_min=2, base_weeks=4,
                  balanced=True, group_jaccard=0.6):
    """Détecte les termes émergents puis regroupe ceux qui décrivent le même événement.

    Critères pour qu'un terme soit retenu une semaine donnée :
      - score z ≥ z_min (part de la semaine nettement au-dessus de la part passée) ;
      - au moins n_min titres cette semaine-là (évite les coïncidences sur 2-3 titres) ;
      - présent dans au moins media_min médias (un événement est couvert par plusieurs rédactions ; écarte les artefacts de format d'un média).
    Un terme est ensuite rattaché à sa semaine de pic (z maximal). Les termes de la même semaine dont les ensembles de titres se
    recouvrent (Jaccard ≥ group_jaccard) sont fusionnés : « raphaël arnault », « arnault », « raphaël » = un seul événement.
    """
    z, p, p0 = emergence_scores(counts, n, pi, base_weeks=base_weeks, balanced=balanced)
    tot = counts.sum(axis=0)
    n_media = (counts > 0).sum(axis=0)
    valid = (tot >= n_min) & (n_media >= media_min) & ~np.isnan(z)
    zv = np.where(valid, z, -np.inf)
    peak_w = zv.argmax(axis=0)
    peak_z = zv[peak_w, np.arange(zv.shape[1])]
    cand = [t for t in np.argsort(-peak_z) if peak_z[t] >= z_min]
    week_idx = {w: i for i, w in enumerate(weeks)}
    row_week = df["week_start"].map(week_idx).values
    Bc = B.tocsc()

    def titles(t, w):
        rows = Bc[:, t].indices
        return set(rows[row_week[rows] == w])

    events, used = [], set()
    for t in cand:
        if t in used:
            continue
        w = peak_w[t]
        ts = titles(t, w)
        members = [t]
        for u in cand:
            if u in used or u == t or peak_w[u] != w:
                continue
            us = titles(u, w)
            if us and len(ts & us) / len(ts | us) >= group_jaccard:
                members.append(u)
        used.update(members)
        lead = max(members, key=lambda m: (peak_z[m], len(terms[m])))
        events.append({"semaine": pd.Timestamp(weeks[w]), "w": w, "terme": terms[lead],
                       "termes associés": ", ".join(terms[m] for m in members if m != lead)[:80],
                       "titres au pic": int(tot[w, lead]), "part avant": p0[w, lead], "part au pic": p[w, lead],
                       "z": peak_z[lead], "médias": int(n_media[w, lead]), "t": lead,
                       "trajectoire": classify_trajectory(p[:, lead], w, p0[w, lead])})
    return pd.DataFrame(events).sort_values("z", ascending=False).reset_index(drop=True), (z, p, p0)
