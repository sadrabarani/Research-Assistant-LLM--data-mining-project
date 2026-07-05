# -*- coding: utf-8 -*-
"""
رابط کاربری پیشرفته با Gradio (امتیاز اضافه - بخش هشتم صورت پروژه).
اجرا: python gradio_app.py
"""
import os

# اگر سیستم پشت VPN/پروکسی باشد (مثلاً برای دسترسی به HuggingFace)، همان پروکسی
# می‌تواند جلوی درخواست خودِ Gradio به localhost را بگیرد و باعث خطای
# "startup-events failed (code 503)" شود. با این تنظیم، ترافیک لوکال از پروکسی معاف می‌شود.
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
os.environ.setdefault("no_proxy", "127.0.0.1,localhost")

import gradio as gr

import rag
import voice_input
import functions
from knowledge_base import build_knowledge_base, get_papers_overview
from vectorstore import get_collection


def kb_status():
    try:
        n = get_collection().count()
        papers = get_papers_overview()
        if n == 0:
            return "⚠️ پایگاه دانش خالی است. دکمه «ساخت پایگاه دانش» را بزنید."
        lines = [f"✅ پایگاه دانش آماده است ({n} chunk)."]
        for pid, title in papers.items():
            lines.append(f"- **{pid}**: {title}")
        return "\n".join(lines)
    except Exception as e:
        return f"خطا در بررسی پایگاه دانش: {e}"


def do_build_kb():
    try:
        build_knowledge_base()
        return kb_status()
    except Exception as e:
        return f"خطا در ساخت پایگاه دانش: {e}"


def chat_fn(message, history, audio):
    query = message
    voice_note = ""
    if audio:
        v = voice_input.voice_to_query(audio)
        query = v["refined_transcript"]
        voice_note = f"🎙️ متن خام: {v['raw_transcript']}\n✍️ متن اصلاح‌شده: {v['refined_transcript']}\n\n"
    try:
        answer = rag.answer(query)
    except Exception as e:
        answer = f"⚠️ خطا: {e}"
    return voice_note + answer


def compare_fn():
    result = functions.compare_results()
    return result["table_markdown"], result.get("chart_path")


def survey_fn():
    return functions.generate_survey()


def gap_fn():
    result = functions.find_research_gap()
    lines = []
    for pid, data in result.items():
        lines.append(f"### {data['title']}")
        lines.append("**محدودیت‌ها:** " + "؛ ".join(data.get("limitations", []) or ["-"]))
        lines.append("**مشکلات حل‌نشده:** " + "؛ ".join(data.get("unsolved_problems", []) or ["-"]))
        lines.append("**پیشنهاد آینده:** " + "؛ ".join(data.get("future_directions", []) or ["-"]))
    return "\n\n".join(lines)


def references_fn():
    result = functions.compare_references()
    return result["table_markdown"]


def limitations_fn():
    return functions.compare_limitations()


with gr.Blocks(title="دستیار پژوهشی هوشمند") as demo:
    gr.Markdown("# 🤖 دستیار پژوهشی هوشمند مبتنی بر LLM")

    with gr.Row():
        status_box = gr.Markdown(kb_status())
        build_btn = gr.Button("🔨 ساخت / بازسازی پایگاه دانش")
    build_btn.click(do_build_kb, outputs=status_box)

    with gr.Tab("💬 گفتگو (متن یا صوت)"):
        chatbot = gr.Chatbot(label="گفتگو", height=450)
        with gr.Row():
            txt = gr.Textbox(placeholder="سوال خود را بنویسید...", scale=4)
            audio_in = gr.Audio(sources=["microphone", "upload"], type="filepath", label="یا سوال صوتی", scale=2)
        send_btn = gr.Button("ارسال")

        def respond(message, history, audio):
            answer = chat_fn(message, history, audio)
            user_shown = message or "🎙️ (پیام صوتی)"
            history = history + [
                {"role": "user", "content": user_shown},
                {"role": "assistant", "content": answer},
            ]
            return history, "", None

        send_btn.click(respond, [txt, chatbot, audio_in], [chatbot, txt, audio_in])
        txt.submit(respond, [txt, chatbot, audio_in], [chatbot, txt, audio_in])

    with gr.Tab("📊 مقایسه نتایج"):
        cmp_btn = gr.Button("مقایسه نتایج تجربی مقالات")
        cmp_table = gr.Markdown()
        cmp_chart = gr.Image(label="نمودار مقایسه")
        cmp_btn.click(compare_fn, outputs=[cmp_table, cmp_chart])

    with gr.Tab("📝 تولید Mini-Survey"):
        survey_btn = gr.Button("تولید مرور مقالات")
        survey_out = gr.Markdown()
        survey_btn.click(survey_fn, outputs=survey_out)

    with gr.Tab("🔍 شکاف‌های پژوهشی"):
        gap_btn = gr.Button("تحلیل محدودیت‌ها / کارهای آینده")
        gap_out = gr.Markdown()
        gap_btn.click(gap_fn, outputs=gap_out)

    with gr.Tab("📚 مقایسه رفرنس‌ها"):
        ref_btn = gr.Button("مقایسه رفرنس‌ها بر اساس سال")
        ref_out = gr.Markdown()
        ref_btn.click(references_fn, outputs=ref_out)

    with gr.Tab("⚖️ مقایسه محدودیت‌ها"):
        lim_btn = gr.Button("مقایسه محدودیت‌های مقالات")
        lim_out = gr.Markdown()
        lim_btn.click(limitations_fn, outputs=lim_out)


if __name__ == "__main__":
    demo.launch()