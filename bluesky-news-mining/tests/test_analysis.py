from pathlib import Path
import tempfile
import unittest
import ast
from contextlib import redirect_stdout
import io
import json
import re
import unicodedata
from unittest.mock import patch

import numpy as np
import pandas as pd

from test_bluesky import load_notebook_definitions

analysis = load_notebook_definitions("00_fonctions_analyse.ipynb", "analysis-definitions")


class AnalysisTests(unittest.TestCase):
    def corpus(self, days=31, spike=True):
        rows = []
        for day, date in enumerate(pd.date_range("2026-01-01", periods=days)):
            event = spike and day >= 28
            for number in range(4 if event else 2):
                text = "manifestation greve transport" if event else "ecole quartier quotidien"
                rows.append({"uri": f"post-{day}-{number}", "author": f"author-{day}-{number}",
                             "date": date, "texte_clean": text, "text_emb": text})
        return pd.DataFrame(rows)

    def test_emergence_abstains_with_one_day(self):
        terms, diagnostics = analysis["detect_emerging_terms"](self.corpus(days=1))
        self.assertTrue(terms.empty)
        self.assertEqual(diagnostics["status"], "insufficient_history")
        self.assertEqual(diagnostics["evaluated_windows"], 0)

    def test_emergence_detects_supported_spike(self):
        df = self.corpus()
        terms, diagnostics = analysis["detect_emerging_terms"](df)
        self.assertEqual(diagnostics["status"], "ok")
        self.assertEqual(diagnostics["evaluated_windows"], 1)
        self.assertIn("manifestation", terms["term"].tolist())
        row = terms.set_index("term").loc["manifestation"]
        self.assertEqual(row.n_posts, 12)
        self.assertEqual(row.n_authors, 12)
        events = analysis["group_emerging_events"](terms, df)
        self.assertEqual(len(events), 1)
        self.assertEqual(events.iloc[0].n_posts, 12)

    def test_no_event_and_single_term_are_supported(self):
        df = self.corpus(spike=False)
        terms, diagnostics = analysis["detect_emerging_terms"](df)
        self.assertEqual(diagnostics["status"], "ok")
        self.assertTrue(terms.empty)
        self.assertTrue(analysis["group_emerging_events"](terms, df).empty)
        spike = self.corpus()
        terms, _ = analysis["detect_emerging_terms"](spike)
        self.assertEqual(len(analysis["group_emerging_events"](terms.head(1), spike)), 0)

    def test_emergence_rejects_single_author_and_keeps_source_zero_baseline_rule(self):
        df = self.corpus()
        df["author"] = "same-author"
        terms, _ = analysis["detect_emerging_terms"](df)
        self.assertTrue(terms.empty)
        sparse_df = self.corpus().iloc[[0, -1]].copy()
        terms, diagnostics = analysis["detect_emerging_terms"](sparse_df, min_docs=1)
        self.assertEqual(diagnostics["status"], "ok")
        self.assertIn("max(reference_posts, 1)", diagnostics["source_zero_baseline_rule"])

    def test_cache_checks_text_not_just_uri(self):
        class Encoder:
            calls = 0

            def encode(self, texts, **kwargs):
                self.calls += 1
                return np.ones((len(texts), 3), dtype=np.float32) / np.sqrt(3)

        encoder = Encoder()
        df = pd.DataFrame({"uri": ["same-uri"], "text_emb": ["premier texte"]})
        with tempfile.TemporaryDirectory() as directory:
            _, first = analysis["cached_embeddings"](df, directory, model=encoder)
            _, second = analysis["cached_embeddings"](df, directory, model=encoder)
            df.loc[0, "text_emb"] = "texte modifie"
            _, third = analysis["cached_embeddings"](df, directory, model=encoder)
            self.assertFalse(first["cache_hit"])
            self.assertTrue(second["cache_hit"])
            self.assertFalse(third["cache_hit"])
            self.assertEqual(encoder.calls, 2)

    def test_kmeans_produces_labels_without_engagement_fields(self):
        df = self.corpus(days=6)
        df.loc[df.index >= 6, "texte_clean"] = "football stade equipe victoire"
        df.loc[df.index >= 6, "text_emb"] = "football stade equipe victoire"
        result, info, diagnostics = analysis["fit_kmeans_topics"](df, n_clusters=2, min_df=2)
        self.assertEqual(len(result), len(df))
        self.assertEqual(diagnostics["actual_k"], 2)
        self.assertEqual(info.posts.sum(), len(df))

    def test_no_overlap_skips_encoder(self):
        class ForbiddenEncoder:
            def encode(self, *args, **kwargs):
                raise AssertionError("No encoding should happen without overlapping dates")

        df = self.corpus(days=1).assign(kmeans=0)
        media = pd.DataFrame({"date": [pd.Timestamp("2025-01-01")], "km": [0], "title_clean": ["un titre"]})
        result, diagnostics = analysis["compare_media_topics"](df, media, ForbiddenEncoder())
        self.assertTrue(result.empty)
        self.assertEqual(diagnostics["status"], "no_date_overlap")

    def test_zero_tfidf_rows_are_included_as_in_source(self):
        df = self.corpus(days=6)
        df.loc[df.index >= 6, "texte_clean"] = "football stade equipe victoire"
        df.loc[0, "texte_clean"] = "motuniquetresrare"
        result, info, diagnostics = analysis["fit_kmeans_topics"](df, n_clusters=2, min_df=2)
        self.assertGreaterEqual(result.iloc[0].kmeans, 0)
        self.assertEqual(diagnostics["zero_vector_posts"], 1)
        self.assertTrue(diagnostics["zero_vectors_included"])
        self.assertEqual(info.posts.sum(), len(df))

    def source_cells(self):
        path = Path(__file__).resolve().parents[2] / "bluesky_topics.ipynb"
        return json.loads(path.read_text(encoding="utf-8"))["cells"]

    def test_lexical_cleaning_matches_original_source(self):
        analysis["source_stopwords"]()
        from nltk.corpus import stopwords

        original = "".join(self.source_cells()[15]["source"])
        nodes = [node for node in ast.parse(original).body if isinstance(node, ast.FunctionDef)
                 or (isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) for t in node.targets)
                     and any(t.id in {"custom", "STOP_NOACC", "STOP_ACC"} for t in node.targets))]
        namespace = {"re": re, "unicodedata": unicodedata, "raw_nltk": stopwords.words("french")}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "source_cleaning", "exec"), namespace)
        for text in ["À Lyon, les élèves manifestent aujourd’hui ! #grève https://exemple.fr",
                     "FRANCE covid19 foo_bar 2026 @a.bsky.social rendez-vous 14h",
                     "L'école, d'une ville : ça bouge — vraiment 😊"]:
            self.assertEqual(analysis["analysis_text"](text), namespace["clean_text"](text))

    def test_kmeans_matches_original_source(self):
        df = self.corpus(days=8)
        df.loc[df.index >= 8, "texte_clean"] = "football stade equipe victoire"
        namespace = {"df": df.copy(), "N_KMEANS": 2, "SEED": 42, "np": np,
                     "pd": pd, "display": lambda *args: None}
        code = "".join(self.source_cells()[24]["source"]).split("# --- accord")[0]
        with redirect_stdout(io.StringIO()):
            exec(compile(code, "source_kmeans", "exec"), namespace)
        result, info, diagnostics = analysis["fit_kmeans_topics"](df, n_clusters=2)
        self.assertEqual(result.kmeans.tolist(), namespace["df"].kmeans.tolist())
        self.assertAlmostEqual(diagnostics["silhouette_euclidean"], namespace["sil"])
        self.assertAlmostEqual(diagnostics["stability_ari"], namespace["ari"])
        self.assertEqual(info.top_terms.tolist(), namespace["km_info"].top_terms.tolist())

    def test_emergence_scores_match_original_source(self):
        df = self.corpus()
        namespace = {"df": df.copy(), "np": np, "pd": pd, "MIN_DOCS": 6,
                     "CountVectorizer": analysis["CountVectorizer"],
                     "MIN_AUTHORS": 2, "W": 3, "BASE": 28, "Z_MIN": 4,
                     "COLLECT_START": df.date.min(), "START": df.date.min(), "END": df.date.max(),
                     "display": lambda *args: None}
        with redirect_stdout(io.StringIO()):
            exec(compile("".join(self.source_cells()[28]["source"]), "source_emergence", "exec"), namespace)
        terms, _ = analysis["detect_emerging_terms"](df)
        pd.testing.assert_frame_equal(terms, namespace["emerg"])

    def test_event_grouping_matches_original_source(self):
        df = self.corpus().assign(likes=1, topic_label="exemple")
        terms, _ = analysis["detect_emerging_terms"](df)
        namespace = {"df": df, "emerg": terms, "np": np, "pd": pd, "display": lambda *args: None}
        with redirect_stdout(io.StringIO()):
            exec(compile("".join(self.source_cells()[29]["source"]), "source_events", "exec"), namespace)
        events = analysis["group_emerging_events"](terms, df)
        original = namespace["events"].rename(columns={"exemples": "examples"})
        pd.testing.assert_frame_equal(events.sort_index(axis=1), original.sort_index(axis=1))

    def test_media_loader_joins_titles_using_article_ids(self):
        topics = pd.DataFrame({"article_id": ["b", "a"], "km": [1, 0], "date": ["2026-01-02"] * 2})
        corpus = pd.DataFrame({"article_id": ["a", "b"], "title_clean": ["Titre A", "Titre B"],
                               "in_analysis": [True, True]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "article_topics.parquet"
            path.touch()
            path.with_name("corpus.parquet").touch()
            with patch.object(analysis["pd"], "read_parquet", side_effect=[topics, corpus]):
                result = analysis["load_media_corpus"](path)
        self.assertEqual(result["title_clean"].tolist(), ["Titre B", "Titre A"])
        self.assertEqual(result["km"].tolist(), [1, 0])

    def test_preparation_preserves_input_and_uses_did(self):
        posts = pd.DataFrame({
            "uri": ["a", "b", "c"], "did": ["first", "first", "second"],
            "created_at": ["2026-11-08T10:00:00Z"] * 3,
            "text": ["Manifestation dans les rues aujourd'hui", "Autre marche dans les rues", "Une reponse personnelle"],
            "clean_text": ["Manifestation dans les rues aujourd'hui", "Autre marche dans les rues", "Une reponse personnelle"],
            "is_reply": [0, 0, 1], "has_link": [0, 0, 0],
        })
        original = posts.copy(deep=True)
        result, _, _ = analysis["prepare_analysis_corpus"](
            posts, min_words=2, min_lexical_words=1, max_posts_per_author_day=1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0].author, "first")
        self.assertNotIn("likes", result.columns)
        pd.testing.assert_frame_equal(posts, original)


if __name__ == "__main__":
    unittest.main()
