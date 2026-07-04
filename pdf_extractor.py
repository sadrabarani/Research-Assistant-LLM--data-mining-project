# -*- coding: utf-8 -*-
"""
استخراج متن از فایل PDF مقالات علمی.

چالش اصلی: بسیاری از مقالات (از جمله هر ۳ مقاله این پروژه) در قالب دو ستونی
(two-column) چاپ شده‌اند. کتابخانه‌های ساده مثل PyPDF2/pdfplumber معمولاً متن
را بر اساس ترتیب خطوط در فایل استخراج می‌کنند که باعث می‌شود متن ستون چپ و
راست با هم قاطی شود (مثلاً یک خط از ستون چپ و خط بعدی از ستون راست).

راه‌حل: از PyMuPDF (fitz) استفاده می‌کنیم که متن هر صفحه را به صورت
«بلوک‌های» مستطیلی (block) با مختصات (x0, y0, x1, y1) برمی‌گرداند. با
دسته‌بندی بلوک‌ها بر اساس مختصات x نسبت به وسط صفحه (ستون چپ / راست) و سپس
مرتب‌سازی هر ستون بر اساس y (بالا به پایین)، ترتیب صحیح خواندن مقاله دو ستونی
بازسازی می‌شود. اگر صفحه تک‌ستونی باشد (مثل صفحه اول/چکیده که معمولاً تمام
عرض صفحه است) به صورت خودکار به عنوان یک ستون واحد در نظر گرفته می‌شود.
"""
import os
import re
import fitz  # PyMuPDF

from config import SECTION_NAMES


def _group_blocks_into_columns(blocks, page_width):
    """بلوک‌های یک صفحه را بر اساس ستون چپ/راست گروه‌بندی می‌کند."""
    mid_x = page_width / 2.0
    left_col, right_col, full_width = [], [], []

    for b in blocks:
        x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4]
        if not text.strip():
            continue
        block_width = x1 - x0
        # اگر بلوک بیش از ۶۵٪ عرض صفحه را بگیرد یعنی تک‌ستونی است (مثل تیتر، چکیده، جدول بزرگ)
        if block_width > 0.65 * page_width:
            full_width.append(b)
        elif (x0 + x1) / 2.0 < mid_x:
            left_col.append(b)
        else:
            right_col.append(b)

    # مرتب‌سازی هر گروه بر اساس موقعیت عمودی
    left_col.sort(key=lambda b: b[1])
    right_col.sort(key=lambda b: b[1])
    full_width.sort(key=lambda b: b[1])

    return left_col, right_col, full_width


def _order_page_blocks(page):
    """
    ترتیب صحیح خواندن بلوک‌های یک صفحه را برمی‌گرداند: ابتدا بلوک‌های
    سراسری بالای صفحه (تیتر/نویسنده) که قبل از شروع دو ستون هستند، سپس
    ستون چپ کامل و در ادامه ستون راست، و در نهایت بلوک‌های سراسری پایین صفحه.
    """
    blocks = page.get_text("blocks")
    page_width = page.rect.width

    left_col, right_col, full_width = _group_blocks_into_columns(blocks, page_width)

    if not left_col and not right_col:
        # صفحه کاملاً تک‌ستونی است
        return [b[4] for b in full_width]

    # بلوک‌های سراسری که بالاتر از شروع ستون‌ها هستند (تیتر مقاله و ...)
    top_y = min([b[1] for b in (left_col + right_col)] or [0])
    header_blocks = [b for b in full_width if b[1] < top_y]
    footer_blocks = [b for b in full_width if b[1] >= top_y]

    ordered = (
        [b[4] for b in header_blocks]
        + [b[4] for b in left_col]
        + [b[4] for b in right_col]
        + [b[4] for b in footer_blocks]
    )
    return ordered


def extract_pdf_metadata_title(pdf_path: str) -> str:
    """تلاش برای گرفتن عنوان از متادیتای فایل PDF (در صورت وجود)."""
    try:
        doc = fitz.open(pdf_path)
        title = (doc.metadata or {}).get("title", "") or ""
        doc.close()
        title = title.strip()
        if len(title) > 8 and not title.lower().endswith((".pdf", ".tex")):
            return title
    except Exception:
        pass
    return ""


def extract_full_text(pdf_path: str) -> str:
    """کل متن یک PDF را با رعایت ترتیب صحیح ستون‌ها استخراج می‌کند."""
    doc = fitz.open(pdf_path)
    pages_text = []
    for page in doc:
        blocks_text = _order_page_blocks(page)
        pages_text.append("\n".join(blocks_text))
    doc.close()
    full_text = "\n\n".join(pages_text)

    # نرمال‌سازی فاصله‌ها/خطوط اضافه
    full_text = re.sub(r"[ \t]+", " ", full_text)
    full_text = re.sub(r"\n{3,}", "\n\n", full_text)
    return full_text.strip()


def split_into_sections(full_text: str) -> dict:
    """
    متن استخراج‌شده را بر اساس عناوین بخش‌های استاندارد یک مقاله علمی
    (Abstract, Introduction, Method, Results, Conclusion, References, ...)
    به بخش‌های جداگانه تقسیم می‌کند. اگر عنوانی پیدا نشود کل متن در بخش
    "Full Text" قرار می‌گیرد.
    """
    # الگو: یک خط که فقط شامل یکی از عناوین بخش (با/بدون شماره) است
    # مثل "3. Method" یا "Method" یا "4.1. Implementation Details"
    pattern = re.compile(
        r"^\s*(?:\d+(?:\.\d+)*\.?\s*)?(" + "|".join(map(re.escape, SECTION_NAMES)) + r")\s*$",
        re.IGNORECASE | re.MULTILINE,
    )

    matches = list(pattern.finditer(full_text))
    if not matches:
        return {"Full Text": full_text}

    sections = {}
    for i, m in enumerate(matches):
        name = m.group(1).strip().title()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        content = full_text[start:end].strip()
        if not content:
            continue
        # اگر بخش تکراری است (مثلا Introduction دوباره در References ذکر شده) به آن اضافه کن
        if name in sections:
            sections[name] += "\n" + content
        else:
            sections[name] = content

    # هر متنی که قبل از اولین بخش شناسایی‌شده آمده (معمولا Title+Authors+Abstract)
    preamble = full_text[: matches[0].start()].strip()
    if preamble:
        sections["Preamble"] = preamble

    return sections


def extract_paper(pdf_path: str) -> dict:
    """خروجی نهایی: متن کامل + بخش‌بندی‌شده + عنوان تخمینی مقاله."""
    full_text = extract_full_text(pdf_path)
    sections = split_into_sections(full_text)

    # عنوان مقاله: اول از متادیتای PDF، در غیر این صورت از چند خط ابتدایی preamble
    title = extract_pdf_metadata_title(pdf_path)
    if not title:
        title_source = sections.get("Preamble", full_text)
        lines = [l.strip() for l in title_source.split("\n") if l.strip()]
        title_lines = []
        for l in lines[:4]:
            if len(l) < 15 and title_lines:
                break
            title_lines.append(l)
            if len(" ".join(title_lines)) > 60:
                break
        title = " ".join(title_lines) if title_lines else os.path.basename(pdf_path)

    return {
        "paper_id": os.path.splitext(os.path.basename(pdf_path))[0],
        "title": title,
        "full_text": full_text,
        "sections": sections,
    }


if __name__ == "__main__":
    import sys
    import json

    path = sys.argv[1]
    result = extract_paper(path)
    print("Title:", result["title"])
    print("Sections found:", list(result["sections"].keys()))
    print("Total chars:", len(result["full_text"]))
