# -*- coding: utf-8 -*-
"""
استراتژی Chunking
==================
رویکرد: «تقسیم آگاه از بخش‌بندی مقاله + پنجره لغزان با هم‌پوشانی»

دلیل انتخاب:
۱. مقالات علمی ساختار معنایی مشخصی دارند (Abstract, Method, Results, ...).
   اگر یک chunk از دو بخش متفاوت (مثلاً انتهای Introduction و ابتدای Method)
   تشکیل شود، embedding آن نامنسجم می‌شود و بازیابی دقیق‌تر (retrieval) را
   خراب می‌کند. به همین دلیل ابتدا متن هر مقاله بر اساس بخش‌ها جدا می‌شود.
۲. هر بخش می‌تواند طولانی باشد (مثلاً Experiments)، بنابراین در داخل هر بخش
   از یک پنجره لغزان با اندازه ثابت (CHUNK_SIZE_WORDS کلمه) و هم‌پوشانی
   (CHUNK_OVERLAP_WORDS کلمه) استفاده می‌شود تا: (الف) اندازه هر chunk برای
   مدل embedding مناسب بماند و (ب) جمله/جدولی که دقیقاً روی مرز دو chunk
   افتاده به طور کامل حداقل در یکی از دو chunk حفظ شود.
۳. متادیتای هر chunk (نام مقاله، نام بخش، شماره chunk) ذخیره می‌شود تا در
   بازیابی بتوان به کاربر گفت «طبق بخش Results مقاله X ...» و همچنین در
   توابعی مثل extract_experimental_results بتوان مستقیم فقط سراغ chunk های
   بخش Results/Experiments رفت.
"""
from config import CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS


def _sliding_window_chunks(text: str, size: int, overlap: int):
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(size - overlap, 1)
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        if not piece:
            break
        chunks.append(" ".join(piece))
        if start + size >= len(words):
            break
    return chunks


def chunk_paper(paper: dict) -> list:
    """
    ورودی: خروجی pdf_extractor.extract_paper (شامل sections)
    خروجی: لیستی از دیکشنری‌ها {id, text, metadata}
    """
    paper_id = paper["paper_id"]
    title = paper["title"]
    chunks = []
    idx = 0
    for section_name, section_text in paper["sections"].items():
        if section_name == "Preamble":
            continue  # فقط عنوان/نویسندگان است، برای پاسخگویی ارزش پایینی دارد
        pieces = _sliding_window_chunks(section_text, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS)
        for p_i, piece in enumerate(pieces):
            chunk_id = f"{paper_id}::{section_name}::{p_i}"
            chunks.append({
                "id": chunk_id,
                "text": piece,
                "metadata": {
                    "paper_id": paper_id,
                    "paper_title": title,
                    "section": section_name,
                    "chunk_index": idx,
                },
            })
            idx += 1
    return chunks
