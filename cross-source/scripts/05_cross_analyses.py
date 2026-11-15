"""Analyses croisées presse × Bluesky. Toutes les sorties sont des agrégats (outputs/tables/*.csv, *.json).

1. Statistiques des deux corpus            5. Émergence : même méthode quotidienne appliquée aux deux sources
2. Composition des thèmes communs          6. Premières mentions à l'heure près pour les grands événements
3. Écho sémantique (paires de documents)   7. Acteurs politiques, cadrages lexicaux, mots distinctifs
4. Synchronie des thèmes partagés          8. Liens partagés sur Bluesky vers la presse
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.textprep import STOP_ACC, norm, strip_accents  # noqa: E402

P, T = ROOT / "data" / "processed", ROOT / "outputs" / "tables"
START, END = pd.Timestamp("2026-08-03"), pd.Timestamp("2026-11-01")
RNG = np.random.default_rng(42)

docs_all = pd.read_parquet(P / "docs.parquet")
emb_all = np.load(P / "embeddings.npy")
sent = pd.read_parquet(P / "sentiment.parquet")
docs_all = docs_all.merge(sent, on="doc_id", how="left")
win = ((docs_all.date >= START) & (docs_all.date <= END)).values
docs, emb = docs_all[win].reset_index(drop=True), emb_all[win]
docs = docs.merge(pd.read_parquet(P / "joint_topics.parquet")[["doc_id", "topic"]], on="doc_id", how="left")
docs["ntext"] = docs["text"].map(norm)

TOPIC_LABELS = {  # read from keywords and sampled documents of each topic (see report, Table 3)
    0: "Protests, high-school movement & curfew", 1: "Football & sport results", 2: "Leisure & everyday life (catch-all)",
    3: "Metropole, transport & regional news", 4: "Local culture & everyday life (catch-all)", 5: "Municipal politics & elections",
    6: "Festivals, concerts & museums", 7: "Fires & firefighters", 8: "Police, arrests & court cases", 9: "Accidents & deaths",
    10: "Shops, bars & looting damage", 11: "Weather forecasts & heat", 12: "Farming, drought & wildfires", 13: "Housing & rents",
    14: "Deals & promotions", 15: "Health & hospitals", 16: "Violence & abuse cases", 17: "Street photography",
    18: "Cycling, running & mobility", 19: "Drownings & family tragedies", 20: "Trafficking & looting trials",
    21: "Climate & environmental health", 22: "Economy & jobs", 23: "Trending-topic bots & esports", 24: "Water, lakes & rivers",
    25: "Mountains & 2030 Winter Games", 26: "Public transport (TCL) & strikes", 27: "E-scooters", 28: "Diseases (Shamonda virus)",
    29: "Boxing & fitness", 30: "Palestine flag (Vénissieux)", 31: "Games & leisure venues", 32: "Drones, surveillance & prisons",
    33: "Bookshops & Decitre sale", 34: "Medical research", 35: "Science & public events", 36: "Esports competitions",
    37: "TV news & media commentary", 38: "LGBT topics & homophobic attack"}
out = {}

# ---------------------------------------------------------------- 1. corpus statistics
def corpus_stats(d):
    g = d.groupby("source")
    return pd.DataFrame({
        "documents": g.size(), "authors_or_outlets": g["author"].nunique(),
        "mean_words": g["text"].apply(lambda s: s.str.split().str.len().mean()).round(1),
        "docs_per_day": (g.size() / ((END - START).days + 1)).round(1),
        "negative_%": (100 * g["label"].apply(lambda s: (s == "negative").mean())).round(1),
        "positive_%": (100 * g["label"].apply(lambda s: (s == "positive").mean())).round(1),
        "mean_polarity": g["polarity"].mean().round(3)})
cs = corpus_stats(docs)
cs.to_csv(T / "x1_corpus_stats.csv")
b = docs[docs.source == "bluesky"]
author_counts = b["author"].value_counts()
out["bluesky_top10_author_share"] = round(float(author_counts.head(10).sum() / len(b)), 3)
out["bluesky_single_post_authors_share"] = round(float((author_counts == 1).mean()), 3)
out["sentiment_tone_spearman_press"] = round(float(stats.spearmanr(docs.loc[docs.source == "press", "polarity"],
                                                                     docs.loc[docs.source == "press", "gdelt_tone"]).statistic), 3)
neg = docs.groupby("source")["label"].apply(lambda s: (s == "negative").sum())
ct = np.array([[neg["press"], (docs.source == "press").sum() - neg["press"]], [neg["bluesky"], (docs.source == "bluesky").sum() - neg["bluesky"]]])
out["negativity_chi2_p"] = float(stats.chi2_contingency(ct)[1])

# ---------------------------------------------------------------- 2. joint topic composition, sentiment, engagement
n_src = docs.source.value_counts()
rows = []
for t, sub in docs[docs.topic >= 0].groupby("topic"):
    n_p, n_b = int((sub.source == "press").sum()), int((sub.source == "bluesky").sum())
    sp, sb = n_p / n_src["press"], n_b / n_src["bluesky"]
    lr = np.log2((sb + 0.5 / n_src["bluesky"]) / (sp + 0.5 / n_src["press"]))
    cat = "Bluesky-dominated" if lr >= np.log2(3) else "press-dominated" if lr <= -np.log2(3) else "shared"
    neg_p = sub.loc[sub.source == "press", "p_neg"].mean() if n_p else np.nan
    neg_b = sub.loc[sub.source == "bluesky", "p_neg"].mean() if n_b else np.nan
    rows.append({"topic": t, "label": TOPIC_LABELS[t], "docs": len(sub), "press": n_p, "bluesky": n_b,
                 "share_press_%": round(100 * sp, 2), "share_bluesky_%": round(100 * sb, 2), "log2_ratio": round(lr, 2),
                 "category": cat, "neg_prob_press": round(neg_p, 3), "neg_prob_bluesky": round(neg_b, 3),
                 "bluesky_median_likes": float(sub.loc[sub.source == "bluesky", "likes"].median()) if n_b else np.nan,
                 "bluesky_mean_likes": round(float(sub.loc[sub.source == "bluesky", "likes"].mean()), 1) if n_b else np.nan})
topics = pd.DataFrame(rows).sort_values("docs", ascending=False)
topics.to_csv(T / "x2_joint_topics_composition.csv", index=False)
docs["category"] = docs["topic"].map(topics.set_index("topic")["category"])
out["topic_category_docs"] = docs.groupby(["category", "source"]).size().unstack(fill_value=0).to_dict()
out["topic_category_counts"] = topics.category.value_counts().to_dict()
# Jensen-Shannon divergence between the two topic distributions (0 = identical agendas, 1 = disjoint)
pp = topics.set_index("topic")["press"] / topics["press"].sum()
pb = topics.set_index("topic")["bluesky"] / topics["bluesky"].sum()
mix = (pp + pb) / 2
js = 0.5 * (stats.entropy(pp, mix, base=2) + stats.entropy(pb, mix, base=2))
out["topic_js_divergence"] = round(float(js), 3)
perm = []
lab_src = docs.loc[docs.topic >= 0, "source"].values
lab_top = docs.loc[docs.topic >= 0, "topic"].values
for _ in range(200):
    s = RNG.permutation(lab_src)
    a = pd.Series(lab_top[s == "press"]).value_counts(normalize=True)
    c = pd.Series(lab_top[s == "bluesky"]).value_counts(normalize=True)
    idx = sorted(set(a.index) | set(c.index)); a, c = a.reindex(idx, fill_value=0), c.reindex(idx, fill_value=0)
    m_ = (a + c) / 2
    perm.append(0.5 * (stats.entropy(a, m_, base=2) + stats.entropy(c, m_, base=2)))
out["topic_js_divergence_null_p95"] = round(float(np.percentile(perm, 95)), 4)

# sentiment gap inside shared topics, with bootstrap CI
gap_rows = []
for t in topics[topics.category == "shared"].query("press >= 40 and bluesky >= 40").topic:
    sub = docs[docs.topic == t]
    a, c = sub.loc[sub.source == "press", "p_neg"].values, sub.loc[sub.source == "bluesky", "p_neg"].values
    boots = [RNG.choice(c, len(c)).mean() - RNG.choice(a, len(a)).mean() for _ in range(1000)]
    gap_rows.append({"topic": t, "label": TOPIC_LABELS[t], "neg_press": a.mean(), "neg_bluesky": c.mean(),
                     "gap": c.mean() - a.mean(), "ci_low": np.percentile(boots, 2.5), "ci_high": np.percentile(boots, 97.5)})
pd.DataFrame(gap_rows).round(3).to_csv(T / "x2b_sentiment_gap_shared_topics.csv", index=False)

# ---------------------------------------------------------------- 3. semantic echo between sources
P_idx, B_idx = np.where(docs.source == "press")[0], np.where(docs.source == "bluesky")[0]
dp = docs.date.values[P_idx].astype("datetime64[D]").astype(int)
db = docs.date.values[B_idx].astype("datetime64[D]").astype(int)
S = emb[P_idx] @ emb[B_idx].T
S = np.where(np.abs(dp[:, None] - db[None, :]) <= 2, S, -1.0)
best_p = S.max(1)                 # each press headline -> closest Bluesky post within +-2 days
best_b = S.max(0)                 # each Bluesky post -> closest press headline within +-2 days
docs.loc[P_idx, "echo_sim"], docs.loc[B_idx, "echo_sim"] = best_p, best_b
ann = pd.read_csv(P / "echo_pairs_annotated.csv")
prec = ann.groupby("bin")["same_story"].agg(["sum", "count", "mean"]).round(2)
prec.to_csv(T / "x3a_echo_precision_annotation.csv")
curve = pd.DataFrame([{"threshold": tau, "press_echoed_%": 100 * (best_p >= tau).mean(), "bluesky_echoed_%": 100 * (best_b >= tau).mean()}
                      for tau in np.round(np.arange(0.5, 0.96, 0.05), 2)]).round(2)
curve.to_csv(T / "x3b_echo_curve.csv", index=False)
TAU = 0.80
echo_topic = (docs[docs.topic >= 0].assign(echo=lambda d: d.echo_sim >= TAU)
              .groupby(["topic", "source"])["echo"].mean().unstack().mul(100).round(1))
echo_topic["label"] = echo_topic.index.map(TOPIC_LABELS)
echo_topic.to_csv(T / "x3c_echo_by_topic.csv")
echo_outlet = (docs.assign(echo=docs.echo_sim >= TAU).groupby("outlet")["echo"].mean().mul(100).round(1))
echo_outlet.to_csv(T / "x3d_echo_by_outlet.csv")
out["echo_press_%_tau80"] = round(float(100 * (best_p >= 0.80).mean()), 1)
out["echo_press_%_tau70"] = round(float(100 * (best_p >= 0.70).mean()), 1)
out["echo_bluesky_%_tau80"] = round(float(100 * (best_b >= 0.80).mean()), 1)
out["echo_bluesky_%_tau70"] = round(float(100 * (best_b >= 0.70).mean()), 1)
# a Bluesky post is a "relay" if near-identical to a headline (sim >= 0.9): mostly re-posted press titles
out["bluesky_relay_%_sim90"] = round(float(100 * (best_b >= 0.90).mean()), 1)
# negative/positive echo: is a press headline more likely to be echoed if negative?
pe = docs.loc[P_idx].assign(echo=best_p >= TAU)
out["echo_rate_by_press_sentiment_%"] = pe.groupby("label")["echo"].mean().mul(100).round(1).to_dict()

# ---------------------------------------------------------------- 4. synchrony of shared topics (daily shares, lagged correlation)
days = pd.date_range(START, END)
daily = docs[docs.topic >= 0].groupby(["source", "date", "topic"]).size()
tot = docs.groupby(["source", "date"]).size()
share = (daily / tot).unstack("topic", fill_value=0)
syn = []
for t in topics[(topics.category == "shared") & (topics.docs >= 150)].topic:
    a = share.loc["press"].reindex(days, fill_value=0)[t].rolling(3, center=True, min_periods=1).mean()
    c = share.loc["bluesky"].reindex(days, fill_value=0)[t].rolling(3, center=True, min_periods=1).mean()
    lags = {lag: a.corr(c.shift(-lag)) for lag in range(-5, 6)}   # lag > 0 : Bluesky lags behind the press
    best = max(lags, key=lambda k: -np.inf if pd.isna(lags[k]) else lags[k])
    syn.append({"topic": t, "label": TOPIC_LABELS[t], "r_lag0": round(lags[0], 3), "best_lag_days": best, "r_best": round(lags[best], 3)})
synchrony = pd.DataFrame(syn).sort_values("r_lag0", ascending=False)
synchrony.to_csv(T / "x4_shared_topics_synchrony.csv", index=False)
vol = docs.groupby(["date", "source"]).size().unstack().reindex(days, fill_value=0)
out["daily_volume_corr"] = round(float(vol["press"].corr(vol["bluesky"])), 3)
vol.to_csv(T / "x4b_daily_volume.csv")
share.loc["press"].reindex(days, fill_value=0).to_csv(T / "x4c_topic_share_press_daily.csv")
share.loc["bluesky"].reindex(days, fill_value=0).to_csv(T / "x4d_topic_share_bluesky_daily.csv")

# ---------------------------------------------------------------- 5. emergence: the same daily method on both sources
W, BASE, Z_MIN, MIN_DOCS, MIN_AUTH = 3, 28, 4.0, 6, 2
EVAL_START = START + pd.Timedelta(days=BASE + W - 1)          # press history starts on 3 Aug -> first evaluable day 2 Sep
STOP_NOACC = {strip_accents(w) for w in STOP_ACC}


def emergence(d, src):
    d = d[d.source == src].reset_index(drop=True)
    cv = CountVectorizer(binary=True, min_df=MIN_DOCS, ngram_range=(1, 2), token_pattern=r"(?u)\b[a-z]{3,}\b", stop_words=sorted(STOP_NOACC))
    X = cv.fit_transform(d["ntext"]).tocsc()
    vocab = np.array(cv.get_feature_names_out())
    all_days = pd.date_range(d.date.min(), END)
    di = d.date.map({x: i for i, x in enumerate(all_days)}).values
    D = len(all_days)
    import scipy.sparse as sp
    Dm = sp.csr_matrix((np.ones(len(d)), (np.arange(len(d)), di)), shape=(len(d), D))
    C = (X.T @ Dm).toarray()
    N = np.bincount(di, minlength=D).astype(float)
    Z = np.full(C.shape, -np.inf); OBS = np.zeros(C.shape)
    for k in range(BASE + W - 1, D):
        b0, b1 = k - W - BASE + 1, k - W + 1
        rate = C[:, b0:b1].sum(1) / max(N[b0:b1].sum(), 1)
        obs = C[:, k - W + 1:k + 1].sum(1); exp = rate * N[k - W + 1:k + 1].sum()
        Z[:, k] = (obs - exp) / np.sqrt(exp + 1); OBS[:, k] = obs
    ev_days = np.where(all_days >= EVAL_START)[0]
    rows = []
    authors = d["author"].values
    for t in range(len(vocab)):
        zs = np.where(OBS[t, ev_days] >= MIN_DOCS, Z[t, ev_days], -np.inf)
        if zs.max() < Z_MIN:
            continue
        k = ev_days[int(zs.argmax())]
        onset = ev_days[int(np.argmax(zs >= Z_MIN))]
        docs_t = X[:, t].indices
        recent = docs_t[(di[docs_t] > k - W) & (di[docs_t] <= k)]
        if len(set(authors[recent])) < MIN_AUTH:
            continue
        rows.append({"term": vocab[t], "peak_date": all_days[k], "onset_date": all_days[onset], "z": float(Z[t, k]),
                     "n_docs": int(OBS[t, k]), "n_authors": len(set(authors[recent]))})
    return pd.DataFrame(rows).sort_values("z", ascending=False).reset_index(drop=True)


em_p, em_b = emergence(docs, "press"), emergence(docs, "bluesky")
em_p.to_csv(T / "x5a_emergence_press_daily.csv", index=False)
em_b.to_csv(T / "x5b_emergence_bluesky_daily.csv", index=False)
shared = em_p.merge(em_b, on="term", suffixes=("_press", "_bluesky"))
shared["peak_lag_days"] = (shared.peak_date_bluesky - shared.peak_date_press).dt.days
shared["onset_lag_days"] = (shared.onset_date_bluesky - shared.onset_date_press).dt.days
shared = shared[shared.peak_lag_days.abs() <= 3]                 # same episode only (peaks at most 3 days apart)
shared.sort_values("z_bluesky", ascending=False).to_csv(T / "x5c_emergence_shared_terms.csv", index=False)
out["emergence_terms"] = {"press": len(em_p), "bluesky": len(em_b), "shared_same_episode": len(shared)}
if len(shared):
    out["emergence_shared_peak_lag"] = {"median": float(shared.peak_lag_days.median()), "mean": round(float(shared.peak_lag_days.mean()), 2),
                                        "bluesky_first": int((shared.peak_lag_days < 0).sum()), "same_day": int((shared.peak_lag_days == 0).sum()),
                                        "press_first": int((shared.peak_lag_days > 0).sum())}
    nz = shared.onset_lag_days[shared.onset_lag_days != 0]
    out["emergence_shared_onset_sign_test_p"] = float(stats.binomtest(int((nz < 0).sum()), len(nz)).pvalue) if len(nz) else None
    out["emergence_shared_onset_lag"] = {"bluesky_first": int((shared.onset_lag_days < 0).sum()), "same_day": int((shared.onset_lag_days == 0).sum()),
                                         "press_first": int((shared.onset_lag_days > 0).sum())}

# ---------------------------------------------------------------- 6. first mentions at hour resolution for major events
EVENTS = [  # (event, regex on the normalised text, start of the search window)
    ("High-school blockades begin", r"\blyceen|\blycee|blocus|blocage", "2026-09-22"),
    ("Vénissieux election annulled", r"venissieux.*(annul|municipales)|(annul|municipales).*venissieux", "2026-09-15"),
    ("Pro-Palestine flag at Vénissieux town hall", r"drapeau palestinien", "2026-09-20"),
    ("17 Oct riots: tourist office fire", r"office (de )?tourisme", "2026-10-17"),
    ("17 Oct riots: looting", r"\bpillag|\bpille", "2026-10-17"),
    ("Curfew in the Presqu'ile", r"couvre feu", "2026-10-18"),
    ("Ban on all protests (24 Oct)", r"(interdi).*(manifest)|(manifest).*(interdi)", "2026-10-22"),
    ("Shamonda virus death", r"shamonda", "2026-10-20"),
    ("Trappes attack", r"trappes", "2026-10-24"),
    ("Deranque family asks for R. Arnault hearing", r"audition", "2026-10-26"),
    ("Ultra-right 'remigration' rally (ban requested)", r"remigration|reemigration", "2026-08-10"),
]
fm = []
for name, rx, start in EVENTS:
    row = {"event": name}
    w = docs_all.assign(ntext=docs_all.text.map(norm))
    w = w[(w.timestamp >= pd.Timestamp(start)) & (w.timestamp < pd.Timestamp(start) + pd.Timedelta(days=4))]
    for src in ("press", "bluesky"):
        hit = w[(w.source == src) & w.ntext.str.contains(rx, regex=True)]
        row[f"first_{src}"] = hit.timestamp.min()
        row[f"n_{src}_4d"] = len(hit)
        row[f"peak_day_{src}"] = hit.timestamp.dt.normalize().value_counts().idxmax() if len(hit) else pd.NaT
    row["bluesky_minus_press_hours"] = round((row["first_bluesky"] - row["first_press"]).total_seconds() / 3600, 1) \
        if pd.notna(row["first_bluesky"]) and pd.notna(row["first_press"]) else np.nan
    fm.append(row)
first = pd.DataFrame(fm)
first.to_csv(T / "x6_first_mentions_hours.csv", index=False)

# ---------------------------------------------------------------- 7. politics: actors, frames, distinctive vocabulary
FAMILIES = {  # party names and national leaders only (unambiguous); local figures are reported separately
    "Radical left (LFI, NPA)": r"\blfi\b|insoumis|melenchon|\bnpa\b",
    "Left (PS, PCF, Place publique)": r"\bps\b|socialiste|\bpcf\b|communiste|glucksmann|olivier faure|hollande",
    "Greens (EELV, Les Écologistes)": r"\beelv\b|ecologistes?\b|les verts|tondelier",
    "Centre / government (Renaissance)": r"renaissance|macron|lecornu|\battal\b|gouvernement",
    "Right (LR)": r"\blr\b|republicains|retailleau|wauquiez",
    "Far right (RN, Reconquête)": r"\brn\b|rassemblement national|le pen|bardella|reconquete|zemmour",
}
LOCAL = {"G. Doucet (mayor, Greens)": r"doucet", "B. Bernard (ex-Metropole president, Greens)": r"bruno bernard",
         "V. Sarselli (Metropole president)": r"sarselli", "J.-M. Aulas (candidate)": r"\baulas\b",
         "R. Arnault (MP, LFI)": r"arnault", "Prefect / préfecture": r"\bprefet|prefecture"}
FRAMES = {
    "Order & security": r"casseur|emeutier|pillard|\bvoyou|racaille|delinquan|ensauvag|laxis|insecurit|forces de l ordre|vandal|degradation|saccag|\bordre public",
    "Far-right threat": r"extreme droite|ultra ?droite|fasci|facho|neo ?nazi|\bnazi|identitaire|remigration",
    "Police repression": r"violences? policieres?|repression|matraqu|lacrymo|nasse|\bflic|\bkeuf|bavure",
    "Social grievance": r"vie chere|precarit|salaire|pouvoir d achat|inegalit|justice sociale|colere|retraite",
    "Radical-left threat": r"ultra ?gauche|extreme gauche|black ?bloc|islamo ?gauch|antifa|wokis",
}


def rate_table(d, patterns):
    rows = []
    for name, rx in patterns.items():
        for src in ("press", "bluesky"):
            hit = d[d.source == src].ntext.str.contains(rx, regex=True).values
            boots = [RNG.choice(hit, len(hit)).mean() for _ in range(1000)]
            rows.append({"category": name, "source": src, "per_1000": 1000 * hit.mean(),
                         "ci_low": 1000 * np.percentile(boots, 2.5), "ci_high": 1000 * np.percentile(boots, 97.5), "docs": int(hit.sum())})
    return pd.DataFrame(rows).round(2)


rate_table(docs, FAMILIES).to_csv(T / "x7a_party_mentions.csv", index=False)
rate_table(docs, LOCAL).to_csv(T / "x7b_local_actor_mentions.csv", index=False)
# frames on the contention subset: protest/police/looting topics or contention keywords, 15 Sep - 1 Nov
CONTENTION_TOPICS = {0, 7, 8, 10, 20}
contention = docs[(docs.date >= "2026-09-15") & (docs.topic.isin(CONTENTION_TOPICS)
                                                 | docs.ntext.str.contains(r"manifest|emeute|pillag|couvre feu|blocage|blocus", regex=True))]
out["contention_subset_docs"] = contention.source.value_counts().to_dict()
ft = rate_table(contention, FRAMES)
ft.to_csv(T / "x7c_frames_contention.csv", index=False)
# frame shares among documents that carry at least one frame
fr = contention.copy()
for name, rx in FRAMES.items():
    fr[name] = fr.ntext.str.contains(rx, regex=True)
framed = fr[fr[list(FRAMES)].any(axis=1)]
out["frame_share_among_framed_docs_%"] = (framed.groupby("source")[list(FRAMES)].mean() * 100).round(1).to_dict(orient="index")
out["framed_docs"] = framed.source.value_counts().to_dict()
# which outlets carry the frames (press) - outlet level
out["frames_by_outlet_per_1000"] = (fr[fr.source == "press"].groupby("outlet")[list(FRAMES)].mean() * 1000).round(1).to_dict(orient="index")


def log_odds(d, min_count=5, a0=500):
    """Log-odds ratio with informative Dirichlet prior (Monroe et al., 2008): Bluesky (z > 0) vs press (z < 0)."""
    cv = CountVectorizer(token_pattern=r"(?u)\b[a-z]{3,}\b", stop_words=sorted(STOP_NOACC), ngram_range=(1, 1))
    X = cv.fit_transform(d.ntext); vocab = np.array(cv.get_feature_names_out())
    yb = np.asarray(X[(d.source == "bluesky").values].sum(0)).ravel(); yp = np.asarray(X[(d.source == "press").values].sum(0)).ravel()
    tot_ = yb + yp; prior = a0 * tot_ / tot_.sum()
    nb, np_ = yb.sum(), yp.sum()
    delta = np.log((yb + prior) / (nb + a0 - yb - prior)) - np.log((yp + prior) / (np_ + a0 - yp - prior))
    z = delta / np.sqrt(1 / (yb + prior) + 1 / (yp + prior))
    keep = tot_ >= min_count
    return pd.DataFrame({"word": vocab[keep], "bluesky": yb[keep], "press": yp[keep], "z": z[keep]}).sort_values("z")


lo_rows = []
for name, sub in [("Contention (protests, riots, police)", contention), ("Municipal politics & elections", docs[docs.topic == 5]),
                  ("High-school protest movement", docs[(docs.date >= "2026-09-15") & docs.ntext.str.contains(r"lycee|lyceen|blocus", regex=True)])]:
    lo = log_odds(sub)
    lo_rows.append({"subset": name, "press_distinctive": ", ".join(lo.head(12).word),
                    "bluesky_distinctive": ", ".join(lo.tail(12).word[::-1])})
pd.DataFrame(lo_rows).to_csv(T / "x7d_distinctive_vocabulary.csv", index=False)

# ---------------------------------------------------------------- 8. links shared on Bluesky
lk = pd.read_parquet(P / "bluesky_lyon_links.parquet")
lk = lk[(lk.date >= START) & (lk.date <= END)]
expl = lk.explode("links").dropna(subset=["links"])
expl["domain"] = expl["links"].map(lambda u: (urlparse(u).hostname or "").lower().removeprefix("www."))
PRESS_DOMAINS = {"leprogres.fr": "Le Progrès", "lyoncapitale.fr": "Lyon Capitale", "lyonmag.com": "LyonMag", "bfmtv.com": "BFM (incl. BFM Lyon)",
                 "rue89lyon.fr": "Rue89Lyon", "tribunedelyon.fr": "Tribune de Lyon", "actu.fr": "Actu.fr"}
NATIONAL = ("lemonde.fr", "liberation.fr", "lefigaro.fr", "francetvinfo.fr", "franceinfo.fr", "20minutes.fr", "leparisien.fr", "huffingtonpost.fr",
            "mediapart.fr", "humanite.fr", "ouest-france.fr", "france3-regions.franceinfo.fr", "radiofrance.fr", "lexpress.fr", "lepoint.fr", "cnews.fr",
            "europe1.fr", "valeursactuelles.com", "marianne.net", "streetpress.com", "reporterre.net", "rmc.bfmtv.com")


def dom_group(dm):
    for k, v in PRESS_DOMAINS.items():
        if dm == k or dm.endswith("." + k):
            return "Lyon outlets studied" if k in ("leprogres.fr", "lyoncapitale.fr", "lyonmag.com") else "Other Lyon/regional outlets"
    if any(dm == n or dm.endswith("." + n) for n in NATIONAL):
        return "National media"
    if dm in ("youtube.com", "youtu.be", "instagram.com", "tiktok.com", "x.com", "twitter.com", "facebook.com", "bsky.app", "twitch.tv"):
        return "Platforms (YouTube, Instagram, X…)"
    return "Other websites"


expl["group"] = expl["domain"].map(dom_group)
out["bluesky_lyon_posts_window"] = int(len(lk))
out["bluesky_posts_with_link_%"] = round(float(100 * (lk["links"].str.len() > 0).mean()), 1)
grp = expl.drop_duplicates(["uri", "group"]).groupby("group").size().sort_values(ascending=False)
grp.to_csv(T / "x8a_link_groups.csv")
top_dom = expl.drop_duplicates(["uri", "domain"]).domain.value_counts().head(25)
top_dom.to_csv(T / "x8b_top_domains.csv")
# accounts that relay the studied outlets
studied = expl[expl.group == "Lyon outlets studied"]
out["posts_linking_studied_outlets"] = int(studied.uri.nunique())
out["accounts_linking_studied_outlets"] = int(studied.author.nunique())
out["top_relayers_share_%"] = round(float(100 * studied.drop_duplicates("uri").author.value_counts().head(3).sum() / max(studied.uri.nunique(), 1)), 1)
# exact URL matches with the press corpus -> sharing delay
def url_key(u):
    p = urlparse(u); return (p.hostname or "").lower().removeprefix("www.") + p.path.rstrip("/")
press_urls = docs_all[docs_all.source == "press"].assign(k=lambda d: d.url.map(url_key)).set_index("k")[["timestamp", "outlet"]]
press_urls = press_urls[~press_urls.index.duplicated()]
studied = studied.assign(k=studied.links.map(url_key)).join(press_urls, on="k", rsuffix="_article")
matched = studied.dropna(subset=["timestamp"]).drop_duplicates("uri")
delay_h = (matched.created_at.dt.tz_convert("UTC").dt.tz_localize(None) - matched.timestamp).dt.total_seconds() / 3600
out["exact_url_matches"] = int(len(matched))
out["exact_url_match_articles"] = int(matched.k.nunique())
if len(matched):
    out["sharing_delay_hours"] = {"median": round(float(delay_h.median()), 1), "q25": round(float(delay_h.quantile(.25)), 1),
                                  "q75": round(float(delay_h.quantile(.75)), 1), "share_before_gdelt_detection_%": round(float(100 * (delay_h < 0).mean()), 1)}
    sh = docs_all[docs_all.source == "press"].assign(k=lambda d: d.url.map(url_key))
    shared_articles = set(matched.k)
    sh["shared_on_bluesky"] = sh.k.isin(shared_articles)
    sh = sh.merge(pd.read_parquet(P / "joint_topics.parquet")[["doc_id", "topic"]], on="doc_id", how="left")
    out["shared_articles_by_outlet"] = sh.groupby("outlet")["shared_on_bluesky"].sum().astype(int).to_dict()
    out["press_articles_shared_negative_%"] = round(float(100 * (sh[sh.shared_on_bluesky].label == "negative").mean()), 1)
    out["press_articles_all_negative_%"] = round(float(100 * (sh.label == "negative").mean()), 1)
    top_shared_topics = sh[sh.shared_on_bluesky].topic.map(TOPIC_LABELS).value_counts().head(8)
    out["shared_articles_top_topics"] = top_shared_topics.to_dict()

json.dump(out, open(T / "x0_summary.json", "w"), indent=2, default=str)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 90)
print(json.dumps(out, indent=1, default=str))
print(cs); print(topics.head(40).to_string(index=False)); print(curve.to_string(index=False)); print(synchrony.to_string(index=False))
print(em_p.head(15).to_string()); print(em_b.head(15).to_string()); print(shared.to_string())
print(first.to_string()); print(pd.read_csv(T / "x7a_party_mentions.csv").to_string()); print(pd.read_csv(T / "x7b_local_actor_mentions.csv").to_string())
print(ft.to_string()); print(pd.read_csv(T / "x7d_distinctive_vocabulary.csv").to_string()); print(grp); print(top_dom.head(15))
print(pd.read_csv(T / "x2b_sentiment_gap_shared_topics.csv").to_string())
