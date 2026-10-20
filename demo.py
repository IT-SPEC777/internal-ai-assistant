"""
Минимальная демонстрация без Docker, без Ollama, без Qdrant.
Показывает гибридный поиск (dense + BM25 + RRF), ACL-фильтрацию и гейт релевантности.
Установка: pip install sentence-transformers rank-bm25
Запуск: python minimal_demo.py (из корня проекта)
"""
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from api.utils import acl_allows, is_relevant, rrf_fuse, tokenize
from ingestion.parse import iter_documents

MIN_DENSE_SCORE = 0.82  # см. README, п. 6, порог нужно калибровать


def main():
    print("[init] loading documents and embeddings...")
    docs = list(iter_documents("data/seed"))
    if not docs:
        print("Ошибка: нет данных в data/seed/. Запустите из корня проекта.")
        return

    model = SentenceTransformer("intfloat/multilingual-e5-small")
    vectors = model.encode(
        ["passage: " + d["text"] for d in docs],
        normalize_embeddings=True, show_progress_bar=False,
    )
    bm25 = BM25Okapi([tokenize(d["text"]) for d in docs])

    def search(query: str, user_groups: list[str], k: int = 3):
        # ACL применяется до ранжирования: недоступные чанки в выдачу не попадают вообще
        allowed = [i for i, d in enumerate(docs) if acl_allows(d["allowed_groups"], user_groups)]
        if not allowed:
            return [], None

        qv = model.encode(["query: " + query], normalize_embeddings=True)[0]
        dense = sorted(
            ({"id": i, "score": float(vectors[i] @ qv)} for i in allowed),
            key=lambda x: x["score"], reverse=True,
        )
        q_tokens = tokenize(query)
        bm_scores = bm25.get_scores(q_tokens) if q_tokens else [0.0] * len(docs)
        sparse = sorted(
            ({"id": i, "score": float(bm_scores[i])} for i in allowed if bm_scores[i] > 0),
            key=lambda x: x["score"], reverse=True,
        )
        top_dense = dense[0]["score"]
        if not is_relevant(dense, sparse, MIN_DENSE_SCORE):
            return [], top_dense
        return [docs[h["id"]] for h in rrf_fuse(dense, sparse)[:k]], top_dense

    def show(title: str, query: str, groups: list[str]):
        print(f"\n--- {title} ---")
        hits, top = search(query, groups)
        print(f"  (top dense score: {top:.3f})" if top is not None else "  (нет доступных документов)")
        if not hits:
            print("  (нет результатов, в проде здесь отказ без вызова LLM)")
        for h in hits:
            print(f"  [{h['source']}] {h['text'][:90]!r}...")

    q = "Какие суточные в командировке по России?"
    print(f"\n=== Запрос: {q} ===")
    show("petrova (all, finance)", q, ["all", "finance"])
    show("sidorov (all), ACL должен отсечь финансы", q, ["all"])
    show("sidorov, публичный документ", "Как оформить пропуск в офис?", ["all"])
    show("sidorov, нерелевантный вопрос", "Какая погода в Москве завтра?", ["all"])


if __name__ == "__main__":
    main()