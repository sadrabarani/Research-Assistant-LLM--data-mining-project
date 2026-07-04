# -*- coding: utf-8 -*-
"""
رابط کاربری ترمینالی (CLI) - حداقل خروجی الزامی پروژه.
اجرا: python cli_app.py
"""
import os
import sys

from knowledge_base import build_knowledge_base, get_papers_overview
from config import PAPERS_DIR
import rag
import voice_input


def ensure_kb_ready():
    from vectorstore import get_collection
    col = get_collection()
    if col.count() == 0:
        print("پایگاه دانش خالی است. در حال ساخت پایگاه دانش از روی مقالات موجود در:", PAPERS_DIR)
        build_knowledge_base()
    else:
        print(f"پایگاه دانش با {col.count()} chunk آماده است.")


def print_banner():
    papers = get_papers_overview()
    print("=" * 70)
    print("دستیار پژوهشی هوشمند مبتنی بر LLM")
    print("مقالات بارگذاری‌شده:")
    for pid, title in papers.items():
        print(f"  - [{pid}] {title}")
    print("دستورات ویژه: 'voice:<مسیر فایل صوتی>' برای پرسش صوتی | 'exit' برای خروج")
    print("=" * 70)


def main():
    ensure_kb_ready()
    print_banner()
    while True:
        try:
            query = input("\nAsk your question:\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nخداحافظ!")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "خروج"):
            print("خداحافظ!")
            break

        if query.startswith("voice:"):
            audio_path = query[len("voice:"):].strip()
            if not os.path.exists(audio_path):
                print(f"فایل صوتی پیدا نشد: {audio_path}")
                continue
            print("در حال تبدیل گفتار به متن...")
            v = voice_input.voice_to_query(audio_path)
            print(f"متن خام Whisper: {v['raw_transcript']}")
            print(f"متن اصلاح‌شده توسط LLM: {v['refined_transcript']}")
            query = v["refined_transcript"]

        print("\nدر حال پردازش...\n")
        try:
            response = rag.answer(query)
        except Exception as e:
            response = f"⚠️ خطا: {e}"
        print(response)


if __name__ == "__main__":
    sys.exit(main())
