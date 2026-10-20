"""Офлайн-тесты чистой логики (только stdlib): python -m unittest discover -s tests -t ."""
import unittest

from api.utils import acl_allows, is_relevant, rrf_fuse, tokenize
from ingestion.parse import chunk_text, iter_documents

USERS = {
    "ivanov": ["all", "it"], "petrova": ["all", "finance"],
    "sidorov": ["all"], "kuznetsov": ["all", "hr"],
}


class TokenizeTests(unittest.TestCase):
    def test_morphology_matches(self):
        self.assertEqual(tokenize("суточные")[0], tokenize("суточных")[0])
        self.assertEqual(tokenize("пропуск")[0], tokenize("пропускной")[0])

    def test_stopwords_removed(self):
        self.assertEqual(tokenize("как в по"), [])

    def test_yo_normalized(self):
        self.assertEqual(tokenize("Пётр"), tokenize("Петр"))


class AclTests(unittest.TestCase):
    def test_allows(self):
        self.assertTrue(acl_allows(["all"], USERS["sidorov"]))
        self.assertTrue(acl_allows(["finance"], USERS["petrova"]))
        self.assertFalse(acl_allows(["finance"], USERS["sidorov"]))
        self.assertFalse(acl_allows(["hr"], USERS["ivanov"]))

    def test_seed_acl_per_user(self):
        docs = list(iter_documents("data/seed"))
        self.assertTrue(docs)
        visible = lambda u: {d["source"] for d in docs if acl_allows(d["allowed_groups"], USERS[u])}
        self.assertEqual(visible("sidorov"), {"public/onboarding.md"})
        self.assertEqual(visible("petrova"), {"public/onboarding.md", "finance/finance_policy.md"})
        self.assertNotIn("finance/finance_policy.md", visible("ivanov"))


class ChunkTests(unittest.TestCase):
    def test_no_empty_and_unique_ids(self):
        docs = list(iter_documents("data/seed"))
        self.assertTrue(all(d["text"].strip() for d in docs))
        ids = [d["id"] for d in docs]
        self.assertEqual(len(ids), len(set(ids)))

    def test_long_text_split(self):
        text = "\n\n".join(f"Абзац {i}. " + "слово " * 40 for i in range(10))
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunks))

    def test_all_content_covered(self):
        text = "Первый абзац.\n\nВторой абзац.\n\nТретий абзац."
        joined = " ".join(chunk_text(text, chunk_size=20, overlap=0))
        for w in ("Первый", "Второй", "Третий"):
            self.assertIn(w, joined)


class FusionTests(unittest.TestCase):
    def test_rrf_prefers_docs_in_both_lists(self):
        dense = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        sparse = [{"id": "b"}, {"id": "d"}]
        fused = rrf_fuse(dense, sparse)
        self.assertEqual(fused[0]["id"], "b")
        self.assertEqual({f["id"] for f in fused}, {"a", "b", "c", "d"})

    def test_rrf_empty(self):
        self.assertEqual(rrf_fuse([], []), [])


class GateTests(unittest.TestCase):
    def test_lexical_hit_passes(self):
        self.assertTrue(is_relevant([{"score": 0.1}], [{"score": 2.0}], 0.82))

    def test_zero_bm25_does_not_pass(self):
        self.assertFalse(is_relevant([{"score": 0.5}], [{"score": 0.0}], 0.82))

    def test_dense_threshold(self):
        self.assertTrue(is_relevant([{"score": 0.9}], [], 0.82))
        self.assertFalse(is_relevant([{"score": 0.78}], [], 0.82))

    def test_nothing(self):
        self.assertFalse(is_relevant([], [], 0.82))


if __name__ == "__main__":
    unittest.main()