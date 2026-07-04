# -*- coding: utf-8 -*-
"""
قابلیت پردازش صوت (بخش ششم صورت پروژه - امتیاز اضافه)
=========================================================
مرحله ۱: تبدیل گفتار به متن با OpenAI Whisper (نسخه لوکال/متن‌باز، بدون نیاز
          به API Key)، که هم انگلیسی و هم فارسی را پشتیبانی می‌کند.
مرحله ۲: چون خروجی خام Whisper گاهی روی اصطلاحات فنی (اسم مقالات، معیارها،
          مثل "RT-DETR"، "ChromaDB"، "mAP") دچار اشتباه می‌شود، متن خام را به
          یک LLM دوم می‌دهیم و از آن می‌خواهیم با در نظر گرفتن اینکه این متن
          یک سوال صوتی درباره مقالات یادگیری عمیق/تشخیص اشیا است، غلط‌های
          احتمالی گفتار-به-متن را اصلاح و متن را روان‌تر کند - بدون اینکه
          معنای اصلی سوال کاربر را تغییر دهد.
"""
import whisper  # pip install openai-whisper

import llm_client
from config import WHISPER_MODEL_SIZE

_model = None


def _load_model():
    global _model
    if _model is None:
        _model = whisper.load_model(WHISPER_MODEL_SIZE)
    return _model


def transcribe_audio(audio_path: str, language: str = None) -> str:
    """language: 'fa', 'en' یا None برای تشخیص خودکار زبان توسط Whisper"""
    model = _load_model()
    result = model.transcribe(audio_path, language=language)
    return result["text"].strip()


REFINEMENT_SYSTEM = (
    "You are a transcription-correction assistant. You receive a raw speech-to-text "
    "transcript of a user's spoken question about computer-vision research papers "
    "(object detection, RT-DETR, YOLO, transformers, COCO dataset, evaluation metrics like "
    "mAP/AP/accuracy/precision/recall/F1/IoU, ChromaDB, embeddings, RAG, etc.). "
    "Fix likely speech-recognition errors (misheard technical terms, broken punctuation, "
    "wrong word splits) WITHOUT changing the user's actual intent or adding new content. "
    "Keep the same language the user spoke in (Persian or English). "
    "Return ONLY the corrected text, nothing else."
)


def refine_transcript(raw_text: str) -> str:
    if not raw_text.strip():
        return raw_text
    prompt = f'Raw transcript:\n"""{raw_text}"""\n\nCorrected transcript:'
    refined = llm_client.ask(prompt, system=REFINEMENT_SYSTEM, temperature=0.0)
    return refined.strip().strip('"')


def voice_to_query(audio_path: str, language: str = None) -> dict:
    """خط لوله کامل: صدا -> متن خام -> متن اصلاح‌شده"""
    raw = transcribe_audio(audio_path, language=language)
    refined = refine_transcript(raw)
    return {"raw_transcript": raw, "refined_transcript": refined}
