"""Эмбеддинг чанков и загрузка в Qdrant. Запуск: python -m ingestion.embed_and_upsert"""
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType
from sentence_transformers import SentenceTransformer

from ingestion.parse import iter_documents
from api.config import settings


def _embed_prefix(model_name: str, is_query: bool, texts: list[str]) -> list[str]:
    """multilingual-e5 требует префиксы query:/passage:."""
    if "e5" in model_name.lower():
        prefix = "query: " if is_query else "passage: "
        return [prefix + t for t in texts]
    return texts


def main():
    docs = list(iter_documents("data/seed"))
    if not docs:
        raise SystemExit("Нет документов в data/seed/. Запускайте из корня проекта.")
    print(f"[init] {len(docs)} chunks found")

    print(f"[init] embedding model: {settings.embed_model}")
    model = SentenceTransformer(settings.embed_model)
    dim = model.get_sentence_embedding_dimension()

    client = QdrantClient(url=settings.qdrant_url)
    if client.collection_exists(settings.qdrant_collection):
        client.delete_collection(settings.qdrant_collection)
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    # индекс по ACL-полю: фильтр применяется быстро и на больших объёмах
    client.create_payload_index(
        collection_name=settings.qdrant_collection,
        field_name="allowed_groups",
        field_schema=PayloadSchemaType.KEYWORD,
    )

    texts = _embed_prefix(settings.embed_model, False, [d["text"] for d in docs])
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    points = [
        PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, d["id"])),
            vector=vec.tolist(),
            payload={
                "text": d["text"],
                "source": d["source"],
                "allowed_groups": d["allowed_groups"],
                "chunk_index": d["chunk_index"],
                "doc_id": d["id"],
            },
        )
        for d, vec in zip(docs, vectors)
    ]
    client.upsert(collection_name=settings.qdrant_collection, points=points)
    print(f"[done] upserted {len(points)} points into '{settings.qdrant_collection}'")


if __name__ == "__main__":
    main()