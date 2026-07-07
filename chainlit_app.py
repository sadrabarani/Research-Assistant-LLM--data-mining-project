# -*- coding: utf-8 -*-
"""
رابط کاربری Chainlit با قابلیت‌های:
- استریم توکن‌به‌توکن پاسخ RAG
- ورودی صوتی (فایل صوتی attach شده به پیام) -> Whisper -> پالایش LLM
- آپلود فایل PDF در چت -> استخراج با PyMuPDF -> استفاده به عنوان context همان سوال
- تاریخچه ساده و خودمختار (SQLite با schema خودمان - بدون وابستگی به
  SQLAlchemyDataLayer رسمی که برای SQLite ناپایدار است و جداول را خودش نمی‌سازد)

اجرا:
    chainlit run chainlit_app.py -w
"""
import os
import sqlite3
from datetime import datetime, timezone

import chainlit as cl

import rag
import functions
import voice_input
from pdf_extractor import extract_paper
from knowledge_base import build_knowledge_base, get_papers_overview
from vectorstore import get_collection

ACTIONS = [
    cl.Action(name="compare_results", payload={}, label="📊 مقایسه نتایج"),
    cl.Action(name="generate_survey", payload={}, label="📝 تولید Mini-Survey"),
    cl.Action(name="find_research_gap", payload={}, label="🔍 شکاف‌های پژوهشی"),
    cl.Action(name="compare_references", payload={}, label="📚 مقایسه رفرنس‌ها"),
    cl.Action(name="compare_limitations", payload={}, label="⚖️ مقایسه محدودیت‌ها"),
    cl.Action(name="show_history", payload={}, label="🕘 تاریخچه پرسش‌ها"),
]

# ------------------------------------------------------------------
# تاریخچه ساده و خودمختار (schema کاملاً در کنترل خودمان، بدون وابستگی
# به داده‌لایه رسمی Chainlit که برای SQLite جداول را خودکار نمی‌سازد)
# ------------------------------------------------------------------
HISTORY_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simple_history.db")


def _init_history_db():
    conn = sqlite3.connect(HISTORY_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def save_history(question: str, answer: str):
    try:
        conn = sqlite3.connect(HISTORY_DB_PATH)
        conn.execute(
            "INSERT INTO history (timestamp, question, answer) VALUES (?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), question, answer),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"⚠️ خطا در ذخیره تاریخچه: {e}")


def load_recent_history(limit: int = 10):
    try:
        conn = sqlite3.connect(HISTORY_DB_PATH)
        rows = conn.execute(
            "SELECT timestamp, question, answer FROM history ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
        return rows
    except Exception:
        return []


def _ensure_kb():
    col = get_collection()
    if col.count() == 0:
        build_knowledge_base()
    return col.count()


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(label="📊 مقایسه نتایج مقالات", message="مقایسه نتایج تجربی مقالات"),
        cl.Starter(label="📝 تولید Mini-Survey", message="یک survey از مقالات تولید کن"),
        cl.Starter(label="🔍 شکاف‌های پژوهشی", message="شکاف‌های پژوهشی و محدودیت‌ها را تحلیل کن"),
        cl.Starter(label="📚 مقایسه رفرنس‌ها", message="رفرنس‌های مقالات را بر اساس سال مقایسه کن"),
    ]


@cl.on_chat_start
async def on_start():
    _init_history_db()
    n = _ensure_kb()
    papers = get_papers_overview()
    lines = [f"✅ پایگاه دانش آماده است ({n} chunk).", "**مقالات:**"]
    for pid, title in papers.items():
        lines.append(f"- **{pid}**: {title}")
    lines.append(
        "\nℹ️ می‌توانید یک فایل PDF (مثلاً مقاله‌ی پنجم) یا فایل صوتی هم به پیام‌تان attach کنید."
    )
    await cl.Message(content="\n".join(lines), actions=ACTIONS).send()


async def _send_followup_actions():
    """پیام جداگانه فقط برای نمایش قابل‌اطمینان دکمه‌های follow-up."""
    await cl.Message(content="", actions=ACTIONS).send()


# ------------------------------------------------------------------
# پردازش فایل‌های ضمیمه‌شده به پیام (صوت / PDF)
# ------------------------------------------------------------------
async def _handle_attachments(message: cl.Message):
    extra_query = None
    extra_context = None

    for el in (message.elements or []):
        mime = getattr(el, "mime", "") or ""
        path = getattr(el, "path", None)
        if not path:
            continue

        if mime.startswith("audio"):
            v = voice_input.voice_to_query(path)
            await cl.Message(
                content=f"🎙️ متن خام Whisper: {v['raw_transcript']}\n✍️ اصلاح‌شده: {v['refined_transcript']}"
            ).send()
            extra_query = v["refined_transcript"]

        elif mime == "application/pdf" or path.lower().endswith(".pdf"):
            await cl.Message(content=f"📄 در حال استخراج متن از `{el.name}` ...").send()
            paper = extract_paper(path)
            preview = "\n\n".join(
                f"[{name}]\n{text[:1500]}" for name, text in paper["sections"].items()
                if name != "Preamble"
            )[:6000]
            extra_context = f'محتوای فایل ضمیمه‌شده "{paper["title"]}":\n{preview}'
            await cl.Message(
                content=f"✅ متن `{el.name}` استخراج شد ({len(paper['sections'])} بخش). "
                        f"می‌توانید درباره‌اش سوال بپرسید."
            ).send()

        elif mime.startswith("image"):
            await cl.Message(
                content=f"⚠️ فایل تصویری `{el.name}` دریافت شد، اما این پروژه فعلاً قابلیت OCR/خواندن "
                        f"محتوای تصویر را ندارد (فقط PDF متنی پشتیبانی می‌شود)."
            ).send()

    return extra_query, extra_context


@cl.on_message
async def on_message(message: cl.Message):
    extra_query, extra_context = await _handle_attachments(message)

    query = extra_query or message.content
    if extra_context and message.content:
        query = f"{extra_context}\n\nسوال کاربر: {message.content}"
    elif extra_context and not message.content:
        query = f"{extra_context}\n\nخلاصه‌ای از این محتوا ارائه بده."

    if not query:
        await _send_followup_actions()
        return

    msg = cl.Message(content="")
    await msg.send()
    full_answer = ""
    try:
        for chunk in rag.answer_stream(query):
            full_answer += chunk
            await msg.stream_token(chunk)
    except Exception as e:
        full_answer = f"⚠️ خطا: {e}"
        await msg.stream_token(full_answer)
    await msg.update()

    save_history(query, full_answer)
    await _send_followup_actions()


# ------------------------------------------------------------------
# Action callback ها
# ------------------------------------------------------------------
import traceback


@cl.action_callback("compare_results")
async def on_compare_results(action: cl.Action):
    try:
        result = functions.compare_results()
        content = "## جدول مقایسه عملکرد\n\n" + result["table_markdown"]
        elements = []
        if result.get("chart_path") and os.path.exists(str(result["chart_path"])):
            elements.append(cl.Image(name="chart", path=result["chart_path"], display="inline"))
        await cl.Message(content=content, elements=elements).send()
    except Exception:
        await cl.Message(content=f"⚠️ خطا در compare_results:\n```\n{traceback.format_exc()}\n```").send()
    await _send_followup_actions()


@cl.action_callback("generate_survey")
async def on_generate_survey(action: cl.Action):
    try:
        survey = functions.generate_survey()
        await cl.Message(content=survey).send()
    except Exception:
        await cl.Message(content=f"⚠️ خطا در generate_survey:\n```\n{traceback.format_exc()}\n```").send()
    await _send_followup_actions()


@cl.action_callback("find_research_gap")
async def on_find_research_gap(action: cl.Action):
    try:
        result = functions.find_research_gap()
        lines = ["## تحلیل محدودیت‌ها و شکاف‌های پژوهشی\n"]
        for pid, data in result.items():
            lines.append(f"### {data['title']}")
            lines.append("**محدودیت‌ها:** " + "؛ ".join(str(x) for x in (data.get("limitations") or ["-"])))
            lines.append("**مشکلات حل‌نشده:** " + "؛ ".join(str(x) for x in (data.get("unsolved_problems") or ["-"])))
            lines.append("**پیشنهاد آینده:** " + "؛ ".join(str(x) for x in (data.get("future_directions") or ["-"])))
        await cl.Message(content="\n\n".join(lines)).send()
    except Exception:
        await cl.Message(content=f"⚠️ خطا در find_research_gap:\n```\n{traceback.format_exc()}\n```").send()
    await _send_followup_actions()


@cl.action_callback("compare_references")
async def on_compare_references(action: cl.Action):
    try:
        result = functions.compare_references()
        await cl.Message(content="## مقایسه رفرنس‌ها بر اساس سال\n\n" + result["table_markdown"]).send()
    except Exception:
        await cl.Message(content=f"⚠️ خطا در compare_references:\n```\n{traceback.format_exc()}\n```").send()
    await _send_followup_actions()


@cl.action_callback("compare_limitations")
async def on_compare_limitations(action: cl.Action):
    try:
        result = functions.compare_limitations()
        await cl.Message(content=result).send()
    except Exception:
        await cl.Message(content=f"⚠️ خطا در compare_limitations:\n```\n{traceback.format_exc()}\n```").send()
    await _send_followup_actions()


@cl.action_callback("show_history")
async def on_show_history(action: cl.Action):
    try:
        rows = load_recent_history(10)
        if not rows:
            await cl.Message(content="هنوز هیچ سوالی ثبت نشده.").send()
        else:
            lines = ["## 🕘 آخرین سوالات\n"]
            for ts, q, a in rows:
                short_a = (a[:200] + "...") if len(a) > 200 else a
                lines.append(f"**[{ts[:19]}] سوال:** {q}\n**پاسخ:** {short_a}\n")
            await cl.Message(content="\n".join(lines)).send()
    except Exception:
        await cl.Message(content=f"⚠️ خطا در show_history:\n```\n{traceback.format_exc()}\n```").send()
    await _send_followup_actions()