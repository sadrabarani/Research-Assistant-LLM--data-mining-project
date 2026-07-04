# -*- coding: utf-8 -*-
"""
ساخت پایگاه دانش مقالات: استخراج متن -> بخش‌بندی -> chunking -> embedding -> ذخیره در ChromaDB
همچنین یک نسخه کامل بخش‌بندی‌شده هر مقاله به صورت JSON در data/kb ذخیره می‌شود
تا توابعی مثل generate_survey / find_research_gap بدون نیاز به جستجوی معنایی
مستقیماً به متن کامل هر بخش (Abstract, Conclusion, ...) دسترسی داشته باشند.
"""
import glob
import json
import os

from config import PAPERS_DIR, KB_DIR
from pdf_extractor import extract_paper
from chunking import chunk_paper
from vectorstore import reset_collection, add_chunks, list_papers


def build_knowledge_base(papers_dir: str = PAPERS_DIR, rebuild: bool = True):
    pdf_files = sorted(glob.glob(os.path.join(papers_dir, "*.pdf")))
    if not pdf_files:
        raise FileNotFoundError(
            f"هیچ فایل PDF ای در {papers_dir} پیدا نشد. حداقل ۴ مقاله را در این پوشه قرار دهید."
        )
    if len(pdf_files) < 4:
        print(f"⚠️  هشدار: طبق صورت پروژه حداقل ۴ مقاله لازم است، فعلاً {len(pdf_files)} مقاله پیدا شد.")

    collection = reset_collection() if rebuild else None

    summary = []
    for pdf_path in pdf_files:
        print(f"در حال پردازش: {pdf_path}")
        paper = extract_paper(pdf_path)

        # ذخیره متادیتای کامل مقاله (برای دسترسی مستقیم بدون RAG)
        kb_path = os.path.join(KB_DIR, f"{paper['paper_id']}.json")
        with open(kb_path, "w", encoding="utf-8") as f:
            json.dump(paper, f, ensure_ascii=False, indent=2)

        # chunking + embedding + ذخیره در chroma
        chunks = chunk_paper(paper)
        add_chunks(chunks, collection=collection)

        summary.append({
            "paper_id": paper["paper_id"],
            "title": paper["title"],
            "sections": list(paper["sections"].keys()),
            "num_chunks": len(chunks),
        })
        print(f"  -> {len(chunks)} chunk تولید و ذخیره شد. بخش‌ها: {list(paper['sections'].keys())}")

    return summary


def load_paper_json(paper_id: str) -> dict:
    path = os.path.join(KB_DIR, f"{paper_id}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_papers() -> list:
    papers = []
    for path in sorted(glob.glob(os.path.join(KB_DIR, "*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            papers.append(json.load(f))
    return papers


def get_papers_overview() -> dict:
    """paper_id -> title، برای نمایش سریع در CLI/Gradio"""
    return list_papers()


if __name__ == "__main__":
    build_knowledge_base()
