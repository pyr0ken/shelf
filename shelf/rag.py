import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import httpx
from fastembed import TextEmbedding

from shelf.config import (
    DEFAULT_EMBEDDING_MODEL,
    MODELS_CACHE_DIR,
    detect_llm_base_url,
    LLM_API_KEY,
    LLM_MODEL
)
from shelf.db import (
    get_book,
    save_chunks,
    get_chunks_with_embeddings,
    search_chunks_fts,
    get_all_books
)
from shelf.scanner import chunk_book_text

_embedding_model: Optional[TextEmbedding] = None

def get_embedding_model() -> TextEmbedding:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = TextEmbedding(
            model_name=DEFAULT_EMBEDDING_MODEL,
            cache_dir=str(MODELS_CACHE_DIR)
        )
    return _embedding_model

def index_book_rag(book_id: int, batch_size: int = 64) -> int:
    book = get_book(book_id)
    if not book:
        return 0

    filepath = Path(book["filepath"])
    toc = json.loads(book["toc_json"]) if book["toc_json"] else None
    chunks = chunk_book_text(filepath, toc=toc)

    if not chunks:
        # No extractable text (e.g. scanned image PDF)
        return 0

    model = get_embedding_model()
    texts = [c["content"] for c in chunks]

    # Generate embeddings in batches
    embeddings_gen = model.embed(texts, batch_size=batch_size)
    for i, emb in enumerate(embeddings_gen):
        chunks[i]["embedding"] = emb

    save_chunks(book_id, chunks)
    return len(chunks)

def vector_search(
    query: str,
    category: Optional[str] = None,
    book_id: Optional[int] = None,
    limit: int = 30
) -> List[Dict[str, Any]]:
    chunks = get_chunks_with_embeddings(category=category, book_id=book_id)
    if not chunks:
        return []

    model = get_embedding_model()
    query_emb = list(model.embed([query]))[0]

    # Stack all chunk embeddings into a 2D numpy matrix
    embeddings_matrix = np.vstack([c["embedding"] for c in chunks])
    
    # Cosine similarity
    query_norm = np.linalg.norm(query_emb)
    matrix_norms = np.linalg.norm(embeddings_matrix, axis=1)
    similarities = np.dot(embeddings_matrix, query_emb) / (matrix_norms * query_norm + 1e-9)

    # Top indices
    top_indices = np.argsort(similarities)[::-1][:limit]
    
    results = []
    for rank, idx in enumerate(top_indices, 1):
        item = dict(chunks[idx])
        item.pop("embedding", None)  # Strip blob
        item["vector_score"] = float(similarities[idx])
        item["vector_rank"] = rank
        results.append(item)
    return results

def search_hybrid(
    query: str,
    category: Optional[str] = None,
    book_id: Optional[int] = None,
    top_k: int = 5,
    rrf_k: int = 60
) -> List[Dict[str, Any]]:
    # 1. Keyword search (FTS5)
    # Sanitize query for FTS5 syntax
    clean_fts_query = "".join(c if c.isalnum() or c.isspace() else " " for c in query).strip()
    fts_results = []
    if clean_fts_query:
        # Match any word or phrase
        fts_terms = " OR ".join(f'"{t}"' for t in clean_fts_query.split() if len(t) > 2)
        if fts_terms:
            try:
                fts_results = search_chunks_fts(fts_terms, category=category, book_id=book_id, limit=30)
            except Exception:
                fts_results = []

    # 2. Dense vector search
    vec_results = vector_search(query, category=category, book_id=book_id, limit=30)

    # 3. Reciprocal Rank Fusion (RRF)
    scores: Dict[int, float] = {}
    items_by_id: Dict[int, Dict[str, Any]] = {}

    for rank, item in enumerate(fts_results, 1):
        cid = item["id"]
        scores[cid] = scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))
        items_by_id[cid] = item

    for rank, item in enumerate(vec_results, 1):
        cid = item["id"]
        scores[cid] = scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))
        if cid not in items_by_id:
            items_by_id[cid] = item

    # Sort by RRF score descending
    sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
    
    top_results = []
    for cid in sorted_ids[:top_k]:
        res = dict(items_by_id[cid])
        res["rrf_score"] = scores[cid]
        top_results.append(res)

    return top_results

def generate_rag_answer(
    question: str,
    category: Optional[str] = None,
    book_id: Optional[int] = None,
    top_k: int = 5,
    use_llm: bool = True,
    base_url: Optional[str] = None,
    api_key: str = LLM_API_KEY,
    model_name: str = LLM_MODEL
) -> Dict[str, Any]:
    sources = search_hybrid(question, category=category, book_id=book_id, top_k=top_k)

    if not sources:
        return {
            "answer": "هیچ بخش مرتبطی در کتاب‌های ایندکس‌شده برای این پرسش یافت نشد. لطفاً مطمئن شوید کتاب‌های مربوطه با دستور `shelf index-rag` ایندکس شده باشند.",
            "sources": [],
            "llm_used": False
        }

    # If LLM synthesis is skipped, return raw grounded passages directly (ultra-fast mode)
    if not use_llm:
        lines = [
            f"⚡ **بازیابی سریع محتوا (بدون پردازش LLM):** {len(sources)} بخش مرتبط از میان ۱۳۰,۰۰۰ قطعه متن پیدا شد:\n"
        ]
        for idx, s in enumerate(sources, 1):
            ch = f" ({s['chapter_title']})" if s.get("chapter_title") else ""
            lines.append(f"**منبع {idx}:** «{s['book_title']}» | صفحه {s['page_number']}{ch}")
        return {
            "answer": "\n".join(lines),
            "sources": sources,
            "llm_used": False
        }

    target_base_url = (base_url or detect_llm_base_url()).rstrip('/')

    # Format context
    context_blocks = []
    for i, s in enumerate(sources, 1):
        chapter = f" | فصل: {s['chapter_title']}" if s.get("chapter_title") else ""
        context_blocks.append(
            f"[منبع {i}: کتاب «{s['book_title']}» | صفحه {s['page_number']}{chapter}]\n"
            f"{s['content']}"
        )
    context_text = "\n\n---\n\n".join(context_blocks)

    system_prompt = (
        "شما یک دستیار هوشمند، دقیق و ارشد فنی برای کتابخانه تخصصی کاربر هستید.\n"
        "خروجی خود را با ساختار غنی و حرفه‌ای Markdown قالب‌بندی کنید:\n"
        "1. از تیترهای شفاف (## و ###) برای تفکیک موضوعات استفاده کنید.\n"
        "2. تمام قطعه‌کدها را حتماً در بلاک‌های کد با مشخص کردن سینتکس دقیق زبان (مانند ```python، ```c، ```cpp، ```bash، ```rust، ```sql) قرار دهید.\n"
        "3. برای درک بهتر، از جدول‌های مقایسه‌ای Markdown، لیست‌های نقطه‌ای و کال‌اوت‌ها (> برای نکات کلیدی و هشدارها) استفاده کنید.\n"
        "4. متن‌های توضیحی را به فارسی روان و دقیق بنویسید، اما اصطلاحات فنی، اسامی توابع و فایل‌ها و کدها را به زبان انگلیسی حفظ کنید.\n"
        "5. برای هر مفهوم یا قطعه‌کد ارائه شده، نام دقیق کتاب و شماره صفحه منبع را قید کنید.\n"
        "6. اگر موضوع در قطعات استخراج‌شده موجود نبود، صادقانه اعلام کنید."
    )

    user_prompt = f"متن منابع استخراج‌شده از کتابخانه:\n\n{context_text}\n\nپرسش کاربر:\n{question}"

    # Query LLM endpoint
    try:
        url = f"{target_base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.2
        }
        trust_env = not ("127.0.0.1" in target_base_url or "localhost" in target_base_url)
        with httpx.Client(timeout=45.0, trust_env=trust_env) as client:
            resp = client.post(url, headers=headers, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                answer = data["choices"][0]["message"]["content"]
                return {
                    "answer": answer,
                    "sources": sources
                }
            else:
                return {
                    "answer": f"خطا در ارتباط با LLM (کد وضعیت {resp.status_code}): {resp.text}\n\nدر ادامه بخش‌های مرتبط مستقیماً آورده شده‌اند:",
                    "sources": sources
                }
    except Exception as e:
        return {
            "answer": f"اتصال به سرور هوش مصنوعی برقرار نشد ({e}). با این حال، دقیق‌ترین بخش‌های استخراج‌شده از کتاب‌ها در زیر آمده است:",
            "sources": sources
        }
