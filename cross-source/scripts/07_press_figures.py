"""Figures « presse seule » du rapport (anglais), à exécuter avec l'environnement de lyon-news-mining.

Usage (depuis textmining/) : lyon-news-mining/.venv/bin/python cross-source/scripts/07_press_figures.py
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
LNM = ROOT / "lyon-news-mining"
OUT = ROOT / "report" / "figures"
sys.path.insert(0, str(LNM))
from src.analysis import emergence as E  # noqa: E402
from src.preprocessing.text_cleaning import add_token_columns  # noqa: E402

TEAL, CORAL, GREY, NAVY = "#1E8A8A", "#E4572E", "#8A94A6", "#14213D"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#5B6B86", "axes.labelcolor": NAVY, "xtick.color": NAVY, "ytick.color": NAVY,
                     "savefig.bbox": "tight", "pdf.fonttype": 42, "legend.frameon": False})
EN = {0: "Roadworks & traffic", 1: "Accidents & everyday incidents", 2: "Violence, arrests & justice", 3: "High-school protests",
      4: "Regional weather alerts", 5: "Local life, associations, schools", 6: "Large wildfires", 7: "Drought, climate & farming",
      8: "Public finance & metro projects", 9: "Culture & festivals (catch-all)", 10: "Municipal politics & elections",
      11: "Local weather forecasts", 12: "Beaujolais & grape harvest", 13: "Shootings (Villeurbanne)"}

# 1. weekly share of the press K-Means themes
share = pd.read_csv(LNM / "outputs/tables/06_themes_parts_hebdo.csv", index_col=0)
share.columns = [int(c.split(" ")[0]) for c in share.columns]
order = [9, 5, 1, 3, 10, 8, 4, 12]
H = share[order].T
cmap = LinearSegmentedColormap.from_list("t", ["#FFFFFF", "#BFE3E0", TEAL, "#0E4F4F"])
fig, ax = plt.subplots(figsize=(7.2, 3.0))
im = ax.imshow(H.values, aspect="auto", cmap=cmap, vmin=0, vmax=25)
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        v = H.values[i, j]; ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7, color="white" if v > 14 else NAVY)
ax.set_yticks(range(len(order))); ax.set_yticklabels([EN[k] for k in order], fontsize=7.5)
ax.set_xticks(range(H.shape[1])); ax.set_xticklabels(list(H.columns), fontsize=7)
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(length=0)
for r in (3, 4): ax.add_patch(plt.Rectangle((-0.5, r - 0.5), H.shape[1], 1, fill=False, ec=CORAL, lw=1.6))
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01); cb.set_label("% of the week's headlines", fontsize=7.5); cb.outline.set_visible(False)
fig.savefig(OUT / "press_themes_week.pdf"); plt.close(fig)

# 2. press events timeline
fam = pd.read_csv(LNM / "outputs/tables/07_tableau_sujets_emergents.csv", parse_dates=["semaine"])
NAME = {"Vœu des Échevins": "Vœu des Échevins (civic ceremony)", "Recours contre les municipales": "Appeal against municipal election",
        "Visite de la commission du CIO": "IOC visit (2030 Winter Games)", "Biennale d'art contemporain": "Contemporary Art Biennale",
        "Mouvement lycéen": "High-school protest movement", "Vacances de la Toussaint": "Autumn school holidays",
        "Festival Lumière : frères Coen": "Lumière film festival (Coen brothers)", "Émeutes du 17 octobre (couvre-feu, pillages)": "17 Oct riots: curfew, looting",
        "Procès Iris et Warren": "Iris & Warren trial", "Affaire Quentin Deranque / R. Arnault": "Deranque case: MP hearing request",
        "Attentat de Trappes et hommages": "Trappes attack & tributes", "Halloween": "Halloween", "Gilets jaunes : acte III (31 oct.)": "Yellow vests ‘Act III’",
        "Salon équestre Equita Lyon": "Equita Lyon horse show"}
fam["name"] = fam["événement"].map(NAME)
fam = fam.sort_values(["semaine", "z"], ascending=[True, False]).reset_index(drop=True)
TR = {"montée durable": (TEAL, "lasting rise"), "pic ponctuel": (CORAL, "one-off spike"), "indéterminé (fin de période)": (GREY, "too close to the end to tell")}
fig, ax = plt.subplots(figsize=(7.2, 3.6)); n = len(fam)
for i, r in fam.iterrows():
    y = n - 1 - i; s = 30 + 14 * r.titres; x = r.semaine + pd.Timedelta(days=3)
    ax.scatter(x, y, s=s, color=TR[r.trajectoire][0], alpha=0.9, edgecolor="white", zorder=3)
    ax.text(x + pd.Timedelta(days=0.8 + np.sqrt(s / np.pi) * 0.11), y, f"{r.titres}", va="center", fontsize=7, color=NAVY)
ax.set_yticks(range(n)); ax.set_yticklabels(fam["name"][::-1], fontsize=7.5)
ax.axvspan(pd.Timestamp("2026-10-12"), pd.Timestamp("2026-10-19"), color=CORAL, alpha=0.07, lw=0)
ax.text(pd.Timestamp("2026-10-12") + pd.Timedelta(days=0.3), n - 0.45, "GDELT gap", color=CORAL, fontsize=7)
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=1)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.tick_params(axis="x", labelsize=7)
ax.set_xlim(pd.Timestamp("2026-09-03"), pd.Timestamp("2026-11-03")); ax.set_ylim(-0.7, n - 0.1)
ax.grid(axis="x", color="#E3E7EE", lw=0.8); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0)
for k, (cc, l) in TR.items(): ax.scatter([], [], color=cc, s=40, label=l)
ax.legend(loc="lower left", fontsize=7, title="Trajectory", title_fontsize=7)
ax.set_xlabel("Week of the peak (number = headlines in the peak week)", fontsize=8)
fig.savefig(OUT / "press_events.pdf"); plt.close(fig)

# 3. frequency vs emergence (press, weekly score)
c = pd.read_parquet(LNM / "data/processed/corpus.parquet"); df = add_token_columns(c[c.in_analysis]).reset_index(drop=True)
df = df[df.media_name.isin(E.MAIN_MEDIA)].reset_index(drop=True)
B, terms = E.binary_matrix(E.term_sets(df.tokens_raw, df.tokens_lemma), min_df=6)
weeks = sorted(df.week_start.unique()); counts, n_ = E.weekly_counts(B, df, weeks); pi = E.media_weights(df)
z, p, p0 = E.emergence_scores(counts, n_, pi); tot = counts.sum(axis=(0, 1)); wk = counts.sum(axis=0)
zmax = np.nanmax(np.where(wk >= 3, z, np.nan), axis=0)
ev, _ = E.detect_events(counts, n_, pi, terms, B, df, weeks); lead = np.array(sorted(set(ev.t)))
fig, ax = plt.subplots(figsize=(4.6, 3.0))
ax.scatter(tot, zmax, s=8, color="#C9D0DA", label="other words")
ax.scatter(tot[lead], zmax[lead], s=14, color=CORAL, label="emerging words")
idx = {t: i for i, t in enumerate(terms)}
for w, lab, dx, dy in [("homme", "“homme” (man)", -30, 22), ("blessé", "“blessé” (injured)", -80, -26),
                       ("couvre-feu", "“couvre-feu” (curfew)", 8, 2), ("lycéen", "“lycéen” (pupil)", -105, -20), ("trappe", "“Trappes”", 8, -2)]:
    i = idx[w]; ax.annotate(lab, (tot[i], zmax[i]), xytext=(dx, dy), textcoords="offset points", fontsize=7, color=NAVY,
                            arrowprops=dict(arrowstyle="-", color=GREY, lw=0.6))
ax.axhline(4, color=GREY, ls="--", lw=0.8); ax.text(5.6, 4.6, "z = 4", fontsize=7, color=GREY)
ax.set_xscale("log"); ax.set_xlabel("headlines containing the word (13 weeks, log)"); ax.set_ylabel("highest weekly z")
ax.set_ylim(-1.5, 27.5); ax.legend(fontsize=7, loc="upper left")
fig.savefig(OUT / "press_freq_vs_emergent.pdf"); plt.close(fig)
print("ok")
