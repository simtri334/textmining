"""Figures du rapport (PDF vectoriels, anglais) à partir des tables agrégées de 05_cross_analyses.py."""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
T, F = ROOT / "outputs" / "tables", ROOT / "outputs" / "figures"
sys.path.insert(0, str(ROOT))
PRESS, BSKY, GREY, CORAL, NAVY = "#1E8A8A", "#1D6FD8", "#8A94A6", "#E4572E", "#14213D"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#5B6B86", "axes.labelcolor": NAVY, "xtick.color": NAVY, "ytick.color": NAVY,
                     "axes.titlesize": 9.5, "axes.titleweight": "bold", "legend.frameon": False, "figure.dpi": 150,
                     "savefig.bbox": "tight", "pdf.fonttype": 42})
S = json.load(open(T / "x0_summary.json"))


def save(fig, name):
    fig.savefig(F / f"{name}.pdf"); fig.savefig(F / f"{name}.png", dpi=200); plt.close(fig)


# ---- F1 daily volumes with key events
vol = pd.read_csv(T / "x4b_daily_volume.csv", index_col=0, parse_dates=True)
fig, ax = plt.subplots(figsize=(7.2, 2.6))
for col, c, lab in [("press", PRESS, "Press headlines (3 outlets)"), ("bluesky", BSKY, "Bluesky posts (Lyon area)")]:
    ax.plot(vol.index, vol[col], color=c, lw=0.6, alpha=0.35)
    ax.plot(vol.index, vol[col].rolling(7, center=True, min_periods=1).mean(), color=c, lw=1.8, label=lab + ", 7-day mean")
for d, lab in [("2026-08-14", "remigration\nrally ban"), ("2026-09-22", "high-school\nblockades"), ("2026-10-17", "17 Oct\nriots"),
               ("2026-10-20", ""), ("2026-10-26", "Trappes")]:
    ax.axvline(pd.Timestamp(d), color=GREY, lw=0.6, ls=":")
    if lab:
        ax.text(pd.Timestamp(d), ax.get_ylim()[1] * 0.97, lab, fontsize=7, color=NAVY, ha="center", va="top")
ax.set_ylabel("documents per day"); ax.legend(loc="lower right", fontsize=7.5)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
save(fig, "f1_daily_volume")

# ---- F2 joint topic composition (log2 ratio)
tp = pd.read_csv(T / "x2_joint_topics_composition.csv")
tp = tp[tp.docs >= 50].sort_values("log2_ratio")
col = tp.category.map({"press-dominated": PRESS, "shared": GREY, "Bluesky-dominated": BSKY})
fig, ax = plt.subplots(figsize=(7.2, 6.2))
ax.barh(tp.label, tp.log2_ratio.clip(-4, 4), color=col)
for y, (v, n) in enumerate(zip(tp.log2_ratio, tp.docs)):
    ax.text(np.clip(v, -4, 4) + (0.08 if v >= 0 else -0.08), y, f"{n}", va="center", ha="left" if v >= 0 else "right", fontsize=6.5, color=NAVY)
for x in (-np.log2(3), np.log2(3)):
    ax.axvline(x, color=GREY, ls="--", lw=0.7)
ax.axvline(0, color=NAVY, lw=0.6)
ax.set_xlabel("log2( share of Bluesky posts / share of press headlines )   —   numbers: documents in the topic")
ax.text(-3.9, len(tp) - 0.6, "over-represented\nin the press", color=PRESS, fontsize=7.5, va="top")
ax.text(3.9, 0.4, "over-represented\non Bluesky", color=BSKY, fontsize=7.5, ha="right")
ax.set_xlim(-4.3, 4.3); ax.tick_params(axis="y", labelsize=7.5)
save(fig, "f2_topic_composition")

# ---- F3 echo: rate vs threshold + annotated precision
cu = pd.read_csv(T / "x3b_echo_curve.csv")
pr = pd.read_csv(T / "x3a_echo_precision_annotation.csv")
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [1.4, 1]})
ax = axes[0]
ax.plot(cu.threshold, cu["press_echoed_%"], marker="o", ms=3, color=PRESS, label="press headlines with a Bluesky match")
ax.plot(cu.threshold, cu["bluesky_echoed_%"], marker="o", ms=3, color=BSKY, label="Bluesky posts with a press match")
ax.axvspan(0.70, 0.80, color=CORAL, alpha=0.08); ax.axvline(0.80, color=CORAL, lw=0.8, ls="--")
ax.set(xlabel="cosine similarity threshold τ (same ±2 days)", ylabel="% of documents", title="(a) Cross-source echo rate")
ax.legend(fontsize=7)
ax = axes[1]
ax.bar(pr["bin"], pr["mean"], color=[GREY, CORAL, CORAL, CORAL], alpha=0.85)
for i, (m, s_, c) in enumerate(zip(pr["mean"], pr["sum"], pr["count"])):
    ax.text(i, m + 0.02, f"{s_}/{c}", ha="center", fontsize=7.5, color=NAVY)
ax.set(ylim=(0, 1), ylabel="share judged 'same story'", title="(b) Manual check of 60 pairs")
ax.tick_params(axis="x", labelsize=7, rotation=15)
save(fig, "f3_echo")

# ---- F4 synchrony of shared topics
sp = pd.read_csv(T / "x4c_topic_share_press_daily.csv", index_col=0, parse_dates=True)
sb = pd.read_csv(T / "x4d_topic_share_bluesky_daily.csv", index_col=0, parse_dates=True)
syn = pd.read_csv(T / "x4_shared_topics_synchrony.csv").set_index("topic")
show = [0, 5, 11]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2), sharey=False)
for ax, t in zip(axes, show):
    ax.plot(sp.index, 100 * sp[str(t)].rolling(3, center=True, min_periods=1).mean(), color=PRESS, lw=1.3, label="press")
    ax.plot(sb.index, 100 * sb[str(t)].rolling(3, center=True, min_periods=1).mean(), color=BSKY, lw=1.3, label="Bluesky")
    short = {0: "Protests, high-school movement\n& curfew", 5: "Municipal politics\n& elections", 11: "Weather forecasts\n& heat"}[t]
    ax.set_title(f"{short}  (r = {syn.loc[t, 'r_lag0']:.2f})", fontsize=8)
    ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
axes[0].set_ylabel("% of the day's documents"); axes[0].legend(fontsize=7)
save(fig, "f4_synchrony")

# ---- F5 emergence shared terms: peak dates press vs Bluesky
sh = pd.read_csv(T / "x5c_emergence_shared_terms.csv", parse_dates=["peak_date_press", "peak_date_bluesky"])
sh = sh.sort_values("peak_date_press")
top = sh[~sh.term.isin(["ile", "presqu", "office", "tourisme", "couvre", "raphael", "arnault"])].head(40)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.3), gridspec_kw={"width_ratios": [2.3, 1]})
ax = axes[0]
for y, r in enumerate(top.itertuples()):
    ax.plot([r.peak_date_press, r.peak_date_bluesky], [y, y], color=GREY, lw=0.8)
    ax.scatter(r.peak_date_press, y, color=PRESS, s=18, zorder=3)
    ax.scatter(r.peak_date_bluesky, y, color=BSKY, s=18, zorder=3, marker="D")
ax.set_yticks(range(len(top))); ax.set_yticklabels(top.term, fontsize=6.5)
ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 15])); ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.scatter([], [], color=PRESS, s=18, label="press peak"); ax.scatter([], [], color=BSKY, marker="D", s=18, label="Bluesky peak")
ax.legend(fontsize=7, loc="upper left"); ax.set_title("(a) Terms emerging in both sources (peaks ≤ 3 days apart)")
ax = axes[1]
lag = sh.peak_lag_days.value_counts().sort_index()
ax.bar(lag.index, lag.values, color=[BSKY if v < 0 else PRESS if v > 0 else GREY for v in lag.index])
ax.set(xlabel="peak lag (days)\n< 0: Bluesky first   > 0: press first", ylabel="shared emerging terms", title="(b) Lag distribution")
save(fig, "f5_emergence_lag")

# ---- F6 political mentions (parties and local actors)
pm = pd.read_csv(T / "x7a_party_mentions.csv"); la = pd.read_csv(T / "x7b_local_actor_mentions.csv")
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
for ax, tab, title in [(axes[0], pm, "(a) Party families and national leaders"), (axes[1], la, "(b) Local actors")]:
    cats = tab.category.unique(); y = np.arange(len(cats))
    for k, (src, c) in enumerate([("press", PRESS), ("bluesky", BSKY)]):
        t = tab[tab.source == src].set_index("category").reindex(cats)
        ax.barh(y + (0.2 if k == 0 else -0.2), t.per_1000, height=0.38, color=c, label="press" if k == 0 else "Bluesky",
                xerr=[t.per_1000 - t.ci_low, t.ci_high - t.per_1000], error_kw={"lw": 0.6, "ecolor": NAVY})
    ax.set_yticks(y); ax.set_yticklabels(cats, fontsize=7); ax.invert_yaxis()
    ax.set(xlabel="documents mentioning, per 1,000", title=title)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, fontsize=7.5, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.05, 1, 1)); save(fig, "f6_political_mentions")

# ---- F7 framing of the contention subset
fr = pd.read_csv(T / "x7c_frames_contention.csv")
fig, ax = plt.subplots(figsize=(5.6, 2.5))
cats = fr.category.unique(); y = np.arange(len(cats))
for k, (src, c) in enumerate([("press", PRESS), ("bluesky", BSKY)]):
    t = fr[fr.source == src].set_index("category").reindex(cats)
    ax.barh(y + (0.2 if k == 0 else -0.2), t.per_1000, height=0.38, color=c, label="press" if k == 0 else "Bluesky",
            xerr=[t.per_1000 - t.ci_low, t.ci_high - t.per_1000], error_kw={"lw": 0.6, "ecolor": NAVY})
ax.set_yticks(y); ax.set_yticklabels(cats); ax.invert_yaxis()
ax.set(xlabel="documents using the frame lexicon, per 1,000 (95% bootstrap CI)")
ax.legend(fontsize=7.5, loc="lower right")
save(fig, "f7_frames")

# ---- F8 sentiment
docs = pd.read_parquet(ROOT / "data" / "processed" / "docs.parquet").merge(pd.read_parquet(ROOT / "data" / "processed" / "sentiment.parquet"), on="doc_id")
docs = docs[(docs.date >= "2026-08-03") & (docs.date <= "2026-11-01")]
gap = pd.read_csv(T / "x2b_sentiment_gap_shared_topics.csv").sort_values("neg_press")
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1, 1.6]})
ax = axes[0]
lab = ["negative", "neutral", "positive"]
for k, (src, c) in enumerate([("press", PRESS), ("bluesky", BSKY)]):
    v = docs[docs.source == src].label.value_counts(normalize=True).reindex(lab) * 100
    ax.bar(np.arange(3) + (-0.2 if k == 0 else 0.2), v, width=0.38, color=c, label="press" if k == 0 else "Bluesky")
    for i, x in enumerate(v):
        ax.text(i + (-0.2 if k == 0 else 0.2), x + 1, f"{x:.0f}", ha="center", fontsize=7)
ax.set_xticks(range(3)); ax.set_xticklabels(lab); ax.set(ylabel="% of documents", title="(a) Sentiment labels"); ax.legend(fontsize=7)
ax = axes[1]
y = np.arange(len(gap))
ax.hlines(y, gap.neg_press, gap.neg_bluesky, color=GREY, lw=1)
ax.scatter(gap.neg_press, y, color=PRESS, s=20, label="press", zorder=3); ax.scatter(gap.neg_bluesky, y, color=BSKY, marker="D", s=20, label="Bluesky", zorder=3)
ax.set_yticks(y); ax.set_yticklabels(gap.label, fontsize=6.8)
ax.set(xlabel="mean probability of 'negative'", title="(b) Same topics, two sources"); ax.legend(fontsize=7, loc="lower right")
fig.tight_layout(); save(fig, "f8_sentiment")

# ---- F9 links from Bluesky to news sites + sharing delay
grp = pd.read_csv(T / "x8a_link_groups.csv", index_col=0).iloc[:, 0]
fig, ax = plt.subplots(figsize=(3.6, 2.2))
grp = grp.sort_values()
ax.barh(grp.index, grp.values, color=[PRESS if "studied" in i else GREY for i in grp.index])
for yy, v in enumerate(grp.values):
    ax.text(v + 60, yy, f"{v:,}", va="center", fontsize=7)
ax.set(xlabel="posts linking to at least one site of the group"); ax.tick_params(axis="y", labelsize=7)
save(fig, "f9_links")
print("figures written")
