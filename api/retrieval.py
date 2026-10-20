import asyncio
import logging
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .config import settings
from .utils import acl_allows, is_relevant, rrf_fuse, tokenize

log = logging.getLogger("retrieval")


def _e5_prefix(name: str, is_query: bool, texts: list[str]) -> list[str]:
    if "e5" in name.lower():
        p = "query: " if is_query else "passage: "
        return [p + t for t in texts]
    return texts


class RetrievalService:
    def __init__(self) -> None:
        self.client = QdrantClient(url=settings.qdrant_url)
        self.embedder = SentenceTransformer(settings.embed_model)
        self._bm25: BM25Okapi | None = None
        self._bm25_docs: list[dict] = []
        self._load_bm25_index()

    def _load_bm25_index(self) -> None:
        if not self.client.collection_exists(settings.qdrant_collection):
            raise RuntimeError(
                f"Коллекция '{settings.qdrant_collection}' не найдена в Qdrant. "
                f"Сначала выполните: python -m ingestion.embed_and_upsert"
            )
        all_points, offset = [], None
        while True:
            points, offset = self.client.scroll(
                collection_name=settings.qdrant_collection,
                limit=256, offset=offset, with_payload=True,
            )
            all_points.extend(points)
            if offset is None:
                break
        self._bm25_docs = [p.payload | {"_id": p.id} for p in all_points]
        if self._bm25_docs:
            self._bm25 = BM25Okapi([tokenize(d["text"]) for d in self._bm25_docs])
        log.info("BM25 index built: %d docs", len(self._bm25_docs))

    def _dense(self, query: str, user_groups: list[str], k: int) -> list[dict]:
        q_vec = self.embedder.encode(
            _e5_prefix(settings.embed_model, True, [query]),
            normalize_embeddings=True, show_progress_bar=False,
        )[0].tolist()
        qfilter = Filter(
            must=[FieldCondition(key="allowed_groups", match=MatchAny(any=user_groups))]
        )
        res = self.client.query_points(
            collection_name=settings.qdrant_collection,
            query=q_vec, query_filter=qfilter, limit=k, with_payload=True,
        ).points
        return [
            {"id": p.id, "score": float(p.score),
             "text": p.payload["text"], "source": p.payload["source"],
             "allowed_groups": p.payload["allowed_groups"],
             "doc_id": p.payload["doc_id"]}
            for p in res
        ]

    def _bm25_search(self, query: str, user_groups: list[str], k: int) -> list[dict]:
        q_tokens = tokenize(query)
        if not self._bm25 or not q_tokens:
            return []
        scores = self._bm25.get_scores(q_tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out = []
        for i in ranked:
            if len(out) >= k or scores[i] <= 0:
                break  # нулевой или отрицательный скор значит нет лексического совпадения
            doc = self._bm25_docs[i]
            if not acl_allows(doc["allowed_groups"], user_groups):
                continue
            out.append({
                "id": doc["_id"], "score": float(scores[i]),
                "text": doc["text"], "source": doc["source"],
                "allowed_groups": doc["allowed_groups"], "doc_id": doc["doc_id"],
            })
        return out

    def _search_sync(self, query: str, user_groups: list[str]) -> list[dict]:
        k = settings.top_k_retrieve
        dense = self._dense(query, user_groups, k)
        sparse = self._bm25_search(query, user_groups, k)

        log.info("retrieval: top_dense=%s top_bm25=%s n_sparse=%d",
                 round(dense[0]["score"], 3) if dense else None,
                 round(sparse[0]["score"], 3) if sparse else None, len(sparse))

        if not is_relevant(dense, sparse, settings.min_dense_score):
            return []
        return rrf_fuse(dense, sparse)[: settings.top_k_final]

    async def search(self, query: str, user_groups: list[str]) -> list[dict]:
        return await asyncio.to_thread(self._search_sync, query, user_groups)


@lru_cache
def get_retrieval() -> RetrievalService:
    return RetrievalService()