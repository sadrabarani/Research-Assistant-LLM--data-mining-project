# -*- coding: utf-8 -*-
"""
قابلیت پردازش صوت (بخش ششم صورت پروژه - امتیاز اضافه)
=========================================================
مرحله ۱: دریافت هر نوع فرمت صوتی و تبدیل خودکار کانال‌ها (استریو/مونو) با pydub
مرحله ۲: تبدیل گفتار به متن با OpenAI Whisper به صورت لوکال
مرحله ۳: اصلاح و روان‌سازی اصطلاحات تخصصی مخدوش‌شده توسط LLM
"""
import os
import torch
import whisper  # pip install openai-whisper
from pydub import AudioSegment  # pip install pydub

import llm_client
from config import WHISPER_MODEL_SIZE

_model = None


def _load_model():
    global _model
    if _model is None:
        # تشخیص خودکار سخت‌افزار برای جلوگیری از خطای حافظه گرافیکی (VRAM)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = whisper.load_model(WHISPER_MODEL_SIZE, device=device)
    return _model


def transcribe_audio(audio_path: str, language: str = None) -> str:
    """تنها وظیفه این تابع تبدیل یک فایل صوتی به متن از طریق ویسپر است."""
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
    """
    خط لوله کامل: پشتیبانی از تمامی فرمت‌ها + مدیریت فایل‌های دو کاناله + اصلاح متون با LLM
    """
    # بارگذاری فایل صوتی بدون وابستگی به پسوند آن
    sound = AudioSegment.from_file(audio_path)

    # سناریوی اول: فایل دو کاناله (Stereo) است
    if sound.channels == 2:
        left_channel, right_channel = sound.split_to_mono()

        left_temp = "left_temp.wav"
        right_temp = "right_temp.wav"
        left_channel.export(left_temp, format="wav")
        right_channel.export(right_temp, format="wav")

        raw_left = transcribe_audio(left_temp, language=language)
        raw_right = transcribe_audio(right_temp, language=language)

        os.remove(left_temp)
        os.remove(right_temp)

        # ترکیب متن دو کانال یا اصلاح مجزای آن‌ها (بسته به نیاز سیستم RAG، ما متن را یکپارچه می‌کنیم)
        combined_raw = f"{raw_left} {raw_right}".strip()
        refined = refine_transcript(combined_raw)

        return {
            "raw_transcript": combined_raw,
            "refined_transcript": refined,
            "details": {"channel_1_raw": raw_left, "channel_2_raw": raw_right, "is_stereo": True}
        }

    # سناریوی دوم: فایل تک کاناله (Mono) است
    else:
        # برای اطمینان از اینکه ویسپر با فرمت‌های عجیب به مشکل نمی‌خورد، یک نسخه wav موقت می‌سازیم
        temp_wav = "mono_temp.wav"
        sound.export(temp_wav, format="wav")

        raw = transcribe_audio(temp_wav, language=language)

        os.remove(temp_wav)

        refined = refine_transcript(raw)
        return {
            "raw_transcript": raw,
            "refined_transcript": refined,
            "details": {"is_stereo": False}
        }