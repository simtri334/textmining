"""Outils de modélisation thématique (étape 5) : TF-IDF, LSA + K-Means, LDA, cohérence NPMI."""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, silhouette_score
from sklearn.preprocessing import normalize


def docs_from_tokens(token_lists) -> list[str]:
    return [" ".join(t) for t in token_lists]


def make_tfidf(docs, min_df=3, max_df=0.4, ngram_range=(1, 1)):
    vec = TfidfVectorizer(min_df=min_df, max_df=max_df, sublinear_tf=True, ngram_range=ngram_range, token_pattern=r"\S+")
    return vec, vec.fit_transform(docs)


def lsa(X, n_components=100, seed=0):
    """Réduit la matrice TF-IDF par SVD tronquée (LSA) puis normalise : K-Means euclidien ≈ K-Means sphérique (cosinus)."""
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    return normalize(svd.fit_transform(X)), svd


def kmeans_scan(Z, ks, seeds=(0, 1, 2), ref_labels=None):
    """Pour chaque k : silhouette (cosinus), stabilité (ARI moyen entre graines), AMI avec une référence externe."""
    rows = []
    for k in ks:
        runs = [KMeans(n_clusters=k, n_init=10, random_state=s).fit(Z).labels_ for s in seeds]
        stab = np.mean([adjusted_rand_score(runs[i], runs[j]) for i in range(len(runs)) for j in range(i + 1, len(runs))])
        row = {"k": k, "silhouette": silhouette_score(Z, runs[0], metric="cosine"), "stabilité (ARI)": stab}
        if ref_labels is not None:
            mask = ref_labels.notna().values
            row["AMI vs rubriques"] = adjusted_mutual_info_score(ref_labels[mask], runs[0][mask])
        rows.append(row)
    return pd.DataFrame(rows).set_index("k")


def top_terms_per_cluster(X, labels, vocab, n=10):
    """Termes à plus fort TF-IDF moyen dans chaque cluster."""
    out = {}
    for c in sorted(set(labels)):
        m = np.asarray(X[labels == c].mean(axis=0)).ravel()
        out[c] = [vocab[i] for i in m.argsort()[::-1][:n]]
    return out


def make_counts(docs, min_df=3, max_df=0.3):
    vec = CountVectorizer(min_df=min_df, max_df=max_df, token_pattern=r"\S+")
    return vec, vec.fit_transform(docs)


def lda_topics(lda, vocab, n=10):
    return {t: [vocab[i] for i in comp.argsort()[::-1][:n]] for t, comp in enumerate(lda.components_)}


def npmi_coherence(topic_words: dict, counts_binary: sparse.csr_matrix, vocab_index: dict, eps=1e-12):
    """Cohérence C_NPMI (niveau document) : moyenne, sur les paires des mots les plus probables d'un topic,
    de NPMI = log(p(a,b)/(p(a)p(b))) / -log(p(a,b)). Valeur dans [-1, 1] ; plus haut = mots du topic qui apparaissent ensemble."""
    D = counts_binary.shape[0]
    df = np.asarray(counts_binary.sum(axis=0)).ravel()
    scores = {}
    for t, words in topic_words.items():
        idx = [vocab_index[w] for w in words]
        sub = counts_binary[:, idx]
        co = (sub.T @ sub).toarray()
        vals = []
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                p_ab, p_a, p_b = (co[a, b] + eps) / D, df[idx[a]] / D, df[idx[b]] / D
                vals.append(np.log(p_ab / (p_a * p_b)) / -np.log(p_ab))
        scores[t] = float(np.mean(vals))
    return scores


def lda_scan(Xc, vocab, ks, seed=0, n_top=10):
    vocab_index = {w: i for i, w in enumerate(vocab)}
    binary = (Xc > 0).astype(np.float32).tocsr()
    rng = np.random.RandomState(seed)
    train = rng.rand(Xc.shape[0]) < 0.8
    rows, models = [], {}
    for k in ks:
        lda = LatentDirichletAllocation(n_components=k, learning_method="batch", max_iter=60, random_state=seed,
                                        doc_topic_prior=1.0 / k, topic_word_prior=0.1).fit(Xc[train])
        coh = npmi_coherence(lda_topics(lda, vocab, n_top), binary, vocab_index)
        rows.append({"k": k, "cohérence NPMI (moyenne)": np.mean(list(coh.values())),
                     "perplexité (test)": lda.perplexity(Xc[~train])})
        models[k] = lda
    return pd.DataFrame(rows).set_index("k"), models
