# -*- coding: utf-8 -*-
"""
هسته اصلی سیستم: تشخیص نیاز به فراخوانی تابع (Function Calling) یا پاسخ‌گویی RAG معمولی،
و رفتار سیستم در نبود اطلاعات مرتبط (بخش هفتم صورت پروژه).
"""
import re

import llm_client
import vectorstore
import functions
from config import SIMILARITY_DISTANCE_THRESHOLD, TOP_K

NOT_FOUND_TEMPLATE = """پاسخ شما در مقالات انتخاب‌شده یافت نشد.
اما با استفاده از مدل زبانی، پاسخ عمومی به شرح زیر است:

{general_answer}"""

# ------------------------------------------------------------------
# Function-calling router
# ------------------------------------------------------------------
FUNCTION_REGISTRY = {
    "extract_experimental_results": functions.extract_experimental_results,
    "compare_results": functions.compare_results,
    "generate_survey": functions.generate_survey,
    "find_research_gap": functions.find_research_gap,
    "compare_references": functions.compare_references,
    "compare_limitations": functions.compare_limitations,
}

# قواعد کلیدواژه‌ای سریع (اولویت اول - سریع و قابل اعتماد)
KEYWORD_RULES = [
    (r"\bsurvey\b|مرور\s*مقال|survey", "generate_survey"),
    (r"research gap|شکاف تحقیقات|محدودیت.*آینده|future work", "find_research_gap"),
    (r"reference|رفرنس|منابع.*سال|references", "compare_references"),
    (r"limitation|محدودیت|نقاط ضعف|weakness", "compare_limitations"),
    (r"compare.*(result|accuracy|performance)|مقایسه.*(نتیج|عملکرد|دقت)|نمودار مقایسه|جدول مقایسه", "compare_results"),
    (r"extract.*result|استخراج نتایج|معیار ارزیابی|metric", "extract_experimental_results"),
]


def route_intent(user_query: str) -> str:
    """
    ابتدا با قواعد کلیدواژه‌ای ساده (سریع، بدون فراخوانی مدل) و در صورت عدم
    تطبیق، با پرسیدن از خود LLM تصمیم می‌گیریم که آیا باید یکی از توابع
    function-calling فراخوانی شود یا پاسخ باید از طریق RAG معمولی داده شود.
    """
    q_lower = user_query.lower()
    for pattern, fn_name in KEYWORD_RULES:
        if re.search(pattern, q_lower):
            return fn_name

    system = (
        "You are a function-calling router for a research-assistant system over 4 papers. "
        "Decide which function best matches the user's request. Answer with STRICT JSON only."
    )
    prompt = f"""
Available functions:
- extract_experimental_results: extract evaluation metrics (accuracy/precision/recall/F1/mAP/IoU) from papers
- compare_results: build a comparison table + chart of papers' performance
- generate_survey: write a mini literature survey covering all papers
- find_research_gap: analyze limitations / future work / unsolved problems
- compare_references: compare papers' references by publication year
- compare_limitations: compare weaknesses of the papers
- none: the question is a normal question-answering request that should be answered via retrieval (RAG),
  not one of the above structured functions.

User request: "{user_query}"

Return JSON: {{"function": "<one of the names above or 'none'>"}}
"""
    try:
        parsed = llm_client.ask_json(prompt, system=system)
        fn_name = parsed.get("function", "none")
        if fn_name in FUNCTION_REGISTRY:
            return fn_name
    except Exception:
        pass
    return "none"


def format_function_result(fn_name: str, result) -> str:
    if fn_name == "extract_experimental_results":
        lines = ["## نتایج استخراج‌شده تجربی\n"]
        for pid, data in result.items():
            lines.append(f"### {data['title']}")
            for m, v in data["metrics"].items():
                lines.append(f"- **{m}**: {v}")
            if data.get("dataset"):
                lines.append(f"- dataset: {data['dataset']}")
            lines.append("")
        return "\n".join(lines)

    if fn_name == "compare_results":
        text = "## جدول مقایسه عملکرد\n\n" + result["table_markdown"]
        if result.get("chart_path"):
            text += f"\n\n📊 نمودار مقایسه در فایل ذخیره شد: `{result['chart_path']}`"
        return text

    if fn_name == "generate_survey":
        return result  # خودش متن markdown کامل survey است

    if fn_name == "find_research_gap":
        lines = ["## تحلیل محدودیت‌ها و شکاف‌های پژوهشی\n"]
        for pid, data in result.items():
            lines.append(f"### {data['title']}")
            lines.append("**محدودیت‌ها:** " + "; ".join(data.get("limitations", []) or ["-"]))
            lines.append("**مشکلات حل‌نشده:** " + "; ".join(data.get("unsolved_problems", []) or ["-"]))
            lines.append("**پیشنهاد پژوهش آینده:** " + "; ".join(data.get("future_directions", []) or ["-"]))
            lines.append("")
        return "\n".join(lines)

    if fn_name == "compare_references":
        return "## مقایسه رفرنس‌ها بر اساس سال انتشار\n\n" + result["table_markdown"]

    if fn_name == "compare_limitations":
        return result

    return str(result)


def call_function(fn_name: str, paper_ids: list = None):
    fn = FUNCTION_REGISTRY[fn_name]
    result = fn(paper_ids) if paper_ids else fn()
    return format_function_result(fn_name, result)


# ------------------------------------------------------------------
# RAG معمولی + رفتار در نبود اطلاعات (بخش هفتم)
# ------------------------------------------------------------------
def rag_answer(user_query: str, top_k: int = TOP_K) -> str:
    hits = vectorstore.query(user_query, top_k=top_k)

    has_relevant_context = bool(hits) and (
        hits[0]["distance"] is None or hits[0]["distance"] <= SIMILARITY_DISTANCE_THRESHOLD
    )

    if not has_relevant_context:
        general = llm_client.ask(
            user_query,
            system="You are a knowledgeable assistant. Answer clearly and concisely.",
        )
        return NOT_FOUND_TEMPLATE.format(general_answer=general)

    context_blocks = []
    for h in hits:
        meta = h["metadata"]
        context_blocks.append(
            f"[Paper: {meta['paper_title']} | Section: {meta['section']}]\n{h['text']}"
        )
    context = "\n\n---\n\n".join(context_blocks)

    system = (
        "You are a research assistant. Answer the user's question using ONLY the provided "
        "context from the papers. Always mention which paper(s) support your answer. "
        "If the context does not fully answer the question, say so honestly."
    )
    prompt = f"""Context from papers:
{context}

Question: {user_query}

Answer (in the same language as the question):"""
    return llm_client.ask(prompt, system=system, temperature=0.2)


def rag_answer_stream(user_query: str, top_k: int = TOP_K):
    """
    نسخه استریم rag_answer: یک generator که تکه‌متن‌ها را همزمان با تولید
    توسط مدل yield می‌کند (برای رابط‌هایی مثل Chainlit که توکن‌به‌توکن نمایش می‌دهند).
    """
    hits = vectorstore.query(user_query, top_k=top_k)

    has_relevant_context = bool(hits) and (
        hits[0]["distance"] is None or hits[0]["distance"] <= SIMILARITY_DISTANCE_THRESHOLD
    )

    if not has_relevant_context:
        yield "پاسخ شما در مقالات انتخاب‌شده یافت نشد.\nاما با استفاده از مدل زبانی، پاسخ عمومی به شرح زیر است:\n\n"
        yield from llm_client.ask_stream(
            user_query,
            system="You are a knowledgeable assistant. Answer clearly and concisely.",
        )
        return

    context_blocks = []
    for h in hits:
        meta = h["metadata"]
        context_blocks.append(
            f"[Paper: {meta['paper_title']} | Section: {meta['section']}]\n{h['text']}"
        )
    context = "\n\n---\n\n".join(context_blocks)

    system = (
        "You are a research assistant. Answer the user's question using ONLY the provided "
        "context from the papers. Always mention which paper(s) support your answer. "
        "If the context does not fully answer the question, say so honestly."
    )
    prompt = f"""Context from papers:
{context}

Question: {user_query}

Answer (in the same language as the question):"""
    yield from llm_client.ask_stream(prompt, system=system, temperature=0.2)


# ------------------------------------------------------------------
# نقطه ورود اصلی
# ------------------------------------------------------------------
def answer(user_query: str) -> str:
    fn_name = route_intent(user_query)
    if fn_name != "none":
        try:
            return call_function(fn_name)
        except Exception as e:
            return f"⚠️ خطا هنگام اجرای تابع `{fn_name}`: {e}"
    return rag_answer(user_query)


def answer_stream(user_query: str):
    """
    نسخه استریم answer(): اگر سوال به یکی از توابع function-calling نیاز داشته
    باشد (که چندین فراخوانی/پردازش دارد و طبیعتاً توکن‌به‌توکن نیست)، کل نتیجه
    یکجا yield می‌شود؛ در غیر این صورت پاسخ RAG واقعاً توکن‌به‌توکن استریم می‌شود.
    """
    fn_name = route_intent(user_query)
    if fn_name != "none":
        try:
            yield call_function(fn_name)
        except Exception as e:
            yield f"⚠️ خطا هنگام اجرای تابع `{fn_name}`: {e}"
        return
    yield from rag_answer_stream(user_query)


def answer_stream(user_query: str, extra_context: str = None):
    """
    نسخه‌ی stream شده - برای واسط‌هایی مثل Chainlit که می‌خواهند توکن به
    توکن پاسخ را نمایش دهند. اگر یکی از توابع function-calling تشخیص داده
    شود، چون خروجی آن (جدول/نمودار) از قبل کامل ساخته می‌شود، یکجا yield
    می‌شود؛ فقط بخش تولید متن آزاد (RAG معمولی) واقعاً token-by-token است.
    """
    if extra_context:
        # کاربر یک فایل (PDF/عکس) در همین پیام attach کرده؛ پاسخ را مستقیم بر
        # اساس محتوای همان فایل (نه پایگاه دانش مقالات) با LLM می‌سازیم.
        system = (
            "You are a helpful assistant. Answer the user's question using the "
            "provided document content below."
        )
        prompt = f"Document content:\n{extra_context[:8000]}\n\nQuestion: {user_query}"
        yield from llm_client.stream_ask(prompt, system=system, temperature=0.2)
        return

    fn_name = route_intent(user_query)
    if fn_name != "none":
        try:
            yield call_function(fn_name)
        except Exception as e:
            yield f"⚠️ خطا هنگام اجرای تابع `{fn_name}`: {e}"
        return

    hits = vectorstore.query(user_query, top_k=TOP_K)
    has_relevant_context = bool(hits) and (
        hits[0]["distance"] is None or hits[0]["distance"] <= SIMILARITY_DISTANCE_THRESHOLD
    )

    if not has_relevant_context:
        yield NOT_FOUND_TEMPLATE.format(general_answer="")
        yield from llm_client.stream_ask(
            user_query,
            system="You are a knowledgeable assistant. Answer clearly and concisely.",
        )
        return

    context_blocks = []
    for h in hits:
        meta = h["metadata"]
        context_blocks.append(
            f"[Paper: {meta['paper_title']} | Section: {meta['section']}]\n{h['text']}"
        )
    context = "\n\n---\n\n".join(context_blocks)

    system = (
        "You are a research assistant. Answer the user's question using ONLY the provided "
        "context from the papers. Always mention which paper(s) support your answer. "
        "If the context does not fully answer the question, say so honestly."
    )
    prompt = f"Context from papers:\n{context}\n\nQuestion: {user_query}\n\nAnswer (in the same language as the question):"
    yield from llm_client.stream_ask(prompt, system=system, temperature=0.2)