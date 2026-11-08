from contextlib import closing
from pathlib import Path
import json
import sqlite3
import tempfile
import unittest

import pandas as pd

def load_notebook_definitions(filename, tag):
    """Charger les fonctions des notebooks sans lancer les étapes interactives."""
    path = Path(__file__).resolve().parents[1] / "notebooks" / filename
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code" and tag in cell["metadata"].get("tags", []):
            exec(compile("".join(cell["source"]), str(path), "exec"), namespace)
    return namespace


database_namespace = load_notebook_definitions("00_fonctions_base.ipynb", "database-definitions")
clean_text = database_namespace["clean_text"]
load_posts = database_namespace["load_posts"]
preprocess_posts = database_namespace["preprocess_posts"]

collector_namespace = load_notebook_definitions("00_collecteur_jetstream.ipynb", "collector-definitions")
create_database = collector_namespace["create_database"]
process_event = collector_namespace["process_event"]
has_external_link = collector_namespace["has_external_link"]


class CollectorTests(unittest.TestCase):
    def test_record_lifecycle_and_language_change(self):
        with closing(sqlite3.connect(":memory:")) as connection, connection:
            create_database(connection)
            event = {"kind": "commit", "did": "did:plc:author", "commit": {
                "collection": "app.bsky.feed.post", "rkey": "one", "operation": "create",
                "record": {"langs": ["fr"], "text": "Texte initial"}}}
            self.assertEqual(process_event(connection, event), 1)
            self.assertEqual(process_event(connection, event), 0)
            event["commit"].update(operation="update", record={"langs": ["fr"], "text": "Texte corrigé"})
            self.assertEqual(process_event(connection, event), 0)
            self.assertEqual(connection.execute("SELECT text FROM posts").fetchone()[0], "Texte corrigé")
            event["commit"]["record"]["langs"] = ["en"]
            process_event(connection, event)
            self.assertEqual(connection.execute("SELECT count(*) FROM posts").fetchone()[0], 0)
            event["commit"]["record"]["langs"] = ["fr"]
            process_event(connection, event)
            event["commit"]["operation"] = "delete"
            process_event(connection, event)
            self.assertEqual(connection.execute("SELECT count(*) FROM posts").fetchone()[0], 0)

    def test_links_in_nested_embed(self):
        self.assertTrue(has_external_link({"embed": {"$type": "app.bsky.embed.recordWithMedia",
            "media": {"$type": "app.bsky.embed.external"}}}))
        self.assertFalse(has_external_link({"embed": {"$type": "app.bsky.embed.images"}}))

    def test_read_only_load_orders_mixed_offsets_by_instant(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "posts.db"
            with closing(sqlite3.connect(path)) as connection, connection:
                create_database(connection)
                for key, date in [("older", "2026-11-02T16:00:00+01:00"), ("newer", "2026-11-02T15:30:00Z")]:
                    connection.execute("INSERT INTO posts VALUES (?, ?, ?, ?, ?, ?, ?)",
                                       (key, "author", date, "Texte", "fr", 0, 0))
            posts = load_posts(path, limit=1)
            self.assertEqual(posts.iloc[0].uri, "newer")
            self.assertEqual(str(posts.created_at.dt.tz), "UTC")
            missing = Path(directory) / "missing.db"
            with self.assertRaises(FileNotFoundError):
                load_posts(missing)
            self.assertFalse(missing.exists())


class PreprocessingTests(unittest.TestCase):
    def test_empty_database_can_be_preprocessed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "posts.db"
            with closing(sqlite3.connect(path)) as connection:
                create_database(connection)
            result = preprocess_posts(load_posts(path))
            self.assertTrue(result.empty)
            self.assertIn("text", result.columns)
            self.assertIn("clean_text", result.columns)

    def test_clean_text_preserves_words_case_and_punctuation(self):
        self.assertEqual(
            clean_text("  Grève demain !\nhttps://exemple.fr/article\twww.exemple.fr   À Lyon.  "),
            "Grève demain ! À Lyon.",
        )

    def test_filter_and_deduplicate_cleaned_text_without_changing_input(self):
        posts = pd.DataFrame({
            "uri": ["first", "duplicate", "case", "english", "undeclared", "short", "multilingual"],
            "langs": ["fr", "fr", "fr", "en", None, "fr", "en,fr"],
            "text": ["Grève demain https://a.fr", "Grève\n demain https://b.fr",
                     "grève demain", "Strike tomorrow", "Grève demain", "ok", "Autre sujet"],
        })
        result = preprocess_posts(posts)
        self.assertEqual(result.uri.tolist(), ["first", "case", "multilingual"])
        self.assertEqual(result.clean_text.tolist(), ["Grève demain", "grève demain", "Autre sujet"])
        self.assertNotIn("clean_text", posts.columns)
        self.assertEqual(len(posts), 7)


if __name__ == "__main__":
    unittest.main()
