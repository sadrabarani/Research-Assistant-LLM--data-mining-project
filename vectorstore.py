# -*- coding: utf-8 -*-
"""
لایه پایگاه داده برداری (Vector DB) با استفاده از ChromaDB.
تولید embedding به صورت کاملاً لوکال با sentence-transformers انجام می‌شود
(بدون نیاز به هیچ API Key ای).
"""
import chromadb
from chromadb.utils import embedding_functions

from config import CHROMA_DIR, EMBEDDING_MODEL_NAME, TOP_K

COLLECTION_NAME = "research_papers"

_embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL_NAME
)


def get_client():
    return chromadb.PersistentClient(path=CHROMA_DIR)


def get_collection(client=None):
    client = client or get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection():
    client = get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    return get_collection(client)


def add_chunks(chunks: list, collection=None):
    """chunks: خروجی chunking.chunk_paper (لیست دیکشنری {id, text, metadata})"""
    collection = collection or get_collection()
    if not chunks:
        return
    # ChromaDB به ازای هر add بهتر است batch شود
    BATCH = 100
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        collection.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )


def query(text: str, top_k: int = TOP_K, where: dict = None, collection=None):
    """جستجوی نزدیک‌ترین chunk ها به یک متن پرسش."""
    collection = collection or get_collection()
    res = collection.query(
        query_texts=[text],
        n_results=top_k,
        where=where,
    )
    hits = []
    if res.get("ids") and res["ids"][0]:
        for i in range(len(res["ids"][0])):
            hits.append({
                "id": res["ids"][0][i],
                "text": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                "distance": res["distances"][0][i] if res.get("distances") else None,
            })
    return hits


def get_all_by_paper_and_section(paper_id: str, section_keywords: list, collection=None):
    """
    تمام chunk های یک مقاله خاص که نام بخش‌شان شامل یکی از section_keywords
    باشد را برمی‌گرداند (برای توابعی مثل استخراج نتایج تجربی که باید دقیقاً
    سراغ بخش Results/Experiments بروند، نه جستجوی معنایی).
    """
    collection = collection or get_collection()
    all_data = collection.get(where={"paper_id": paper_id})
    matched = []
    for doc, meta in zip(all_data["documents"], all_data["metadatas"]):
        section = meta.get("section", "")
        if any(kw.lower() in section.lower() for kw in section_keywords):
            matched.append({"text": doc, "metadata": meta})
    return matched


def list_papers(collection=None):
    collection = collection or get_collection()
    all_data = collection.get()
    papers = {}
    for meta in all_data["metadatas"]:
        papers[meta["paper_id"]] = meta["paper_title"]
    return papers
