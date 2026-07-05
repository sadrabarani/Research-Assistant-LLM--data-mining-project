# -*- coding: utf-8 -*-
"""
پیاده‌سازی توابع (Function Calling) طبق بخش چهارم و پنجم صورت پروژه:
الزامی: extract_experimental_results, compare_results, generate_survey
اختیاری: find_research_gap, compare_references, compare_limitations
"""
import json
import os
import re
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import llm_client
from knowledge_base import load_all_papers
from config import OUTPUTS_DIR

MAX_SECTION_CHARS = 6000  # جلوگیری از رد شدن از ظرفیت context مدل لوکال


def _get_section(paper: dict, keywords: list, prioritize_end: bool = False) -> str:
    """متن یک یا چند بخش مرتبط را برمی‌گرداند. اگر prioritize_end فعال باشد، انتهای متن نگه داشته می‌شود."""
    parts = []
    for name, text in paper["sections"].items():
        if any(kw.lower() in name.lower() for kw in keywords):
            parts.append(text)
    combined = "\n".join(parts)

    if len(combined) > MAX_SECTION_CHARS:
        if prioritize_end:
            return combined[-MAX_SECTION_CHARS:]
        return combined[:MAX_SECTION_CHARS]
    return combined


COMMON_METRICS = ["accuracy", "precision", "recall", "f1", "f1 score", "map", "iou", "ap"]

# ---------------------------------------------------------------------------
# تابع اول (الزامی): extract_experimental_results
# ---------------------------------------------------------------------------
def extract_experimental_results(paper_ids: list = None) -> dict:
    """
    برای هر مقاله، از بخش Results/Experiments، حداقل دو معیار ارزیابی مشترک
    (accuracy/precision/recall/f1/map/iou) را با کمک LLM استخراج می‌کند.
    خروجی: {paper_id: {"title":..., "metrics": {metric_name: value_str, ...}, "notes": ...}}
    """
    papers = load_all_papers()
    if paper_ids:
        papers = [p for p in papers if p["paper_id"] in paper_ids]

    results = {}
    for paper in papers:
        # فعال کردن prioritize_end برای بخش نتایج
        section_text = _get_section(paper, ["Experiment", "Result", "Evaluation"], prioritize_end=True)
        if not section_text:
            section_text = _get_section(paper, ["Abstract"])

        system = (
            "You are a research assistant that extracts quantitative evaluation results "
            "from a computer-science paper's experiments section. Always answer in strict JSON."
        )
        prompt = f"""
From the following text (Experiments/Results section of a paper titled "{paper['title']}"),
extract the MAIN reported evaluation metrics (choose from: Accuracy, Precision, Recall, F1 Score, mAP, AP, IoU,
or any other clearly numeric evaluation metric explicitly reported). Include the best/headline
numbers reported by the paper's own proposed method (not just baselines).

Return STRICT JSON with this schema:
{{
  "metrics": {{"<metric_name>": "<value with unit/context, e.g. '54.6 AP on COCO val2017 (R101 backbone)'>"}},
  "dataset": "<main benchmark dataset used>",
  "notes": "<one short sentence summary>"
}}
At least TWO metrics must be included if the text supports it.

TEXT:
{section_text}
"""
        parsed = llm_client.ask_json(prompt, system=system)
        results[paper["paper_id"]] = {
            "title": paper["title"],
            "metrics": parsed.get("metrics", {}),
            "dataset": parsed.get("dataset", ""),
            "notes": parsed.get("notes", ""),
        }
    return results


# ---------------------------------------------------------------------------
# تابع دوم (الزامی): compare_results
# ---------------------------------------------------------------------------
def compare_results(paper_ids: list = None) -> dict:
    """
    از خروجی extract_experimental_results یک جدول مقایسه‌ای (markdown) و
    یک نمودار میله‌ای (png) تولید می‌کند.
    """
    extracted = extract_experimental_results(paper_ids)

    metric_counter = Counter()
    for pid, data in extracted.items():
        for metric_name in data["metrics"]:
            metric_counter[metric_name.strip().lower()] += 1
    common_metrics = [m for m, c in metric_counter.items() if c >= 2]
    if not common_metrics and metric_counter:
        common_metrics = [metric_counter.most_common(1)[0][0]]

    headers = ["Paper"] + [m.upper() for m in common_metrics] if common_metrics else ["Paper", "Metrics"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    chart_data = defaultdict(dict)

    for pid, data in extracted.items():
        row = [data["title"][:40]]
        metrics_lower = {k.strip().lower(): v for k, v in data["metrics"].items()}
        if common_metrics:
            for m in common_metrics:
                val = metrics_lower.get(m, "-")
                row.append(str(val))
                num = _extract_first_number(str(val))
                if num is not None:
                    chart_data[m][data["title"][:25]] = num
        else:
            row.append(json.dumps(data["metrics"], ensure_ascii=False))
        lines.append("| " + " | ".join(row) + " |")

    table_md = "\n".join(lines)
    chart_path = None
    plot_metrics = {m: vals for m, vals in chart_data.items() if len(vals) >= 2}

    if plot_metrics:
        # ایجاد پوشه در صورت عدم وجود
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        n = len(plot_metrics)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
        if n == 1:
            axes = [axes]
        for ax, (metric, vals) in zip(axes, plot_metrics.items()):
            names = list(vals.keys())
            values = list(vals.values())
            ax.bar(names, values, color="#4C72B0")
            ax.set_title(metric.upper())
            ax.set_ylabel("Value")
            ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        chart_path = os.path.join(OUTPUTS_DIR, "compare_results_chart.png")
        plt.savefig(chart_path, dpi=150)
        plt.close(fig)
    else:
        chart_path = "داده عددی معتبری برای رسم نمودار یافت نشد."

    return {"table_markdown": table_md, "chart_path": chart_path, "raw": extracted}

def _extract_first_number(text: str):
    m = re.search(r"(\d+(?:\.\d+)?)", text)
    return float(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# تابع سوم (الزامی): generate_survey
# ---------------------------------------------------------------------------
def generate_survey(paper_ids: list = None) -> str:
    papers = load_all_papers()
    if paper_ids:
        papers = [p for p in papers if p["paper_id"] in paper_ids]

    paper_summaries = []
    for p in papers:
        abstract = _get_section(p, ["Abstract"])
        method = _get_section(p, ["Method", "Methodology", "Approach"])
        results = _get_section(p, ["Result", "Experiment"])
        conclusion = _get_section(p, ["Conclusion"])
        paper_summaries.append(
            f"### Paper: {p['title']} (id={p['paper_id']})\n"
            f"Abstract: {abstract[:1500]}\n"
            f"Method: {method[:1500]}\n"
            f"Results: {results[:1000]}\n"
            f"Conclusion: {conclusion[:800]}\n"
        )

    system = (
        "You are an expert academic writer. Write a well-structured mini-survey in English, "
        "using markdown headings, based ONLY on the information provided about the given papers."
    )
    prompt = f"""
Based on the following {len(papers)} papers, write a Mini-Survey with EXACTLY these markdown sections:

## 1. Problem Introduction
## 2. Method of Each Paper
(one subsection per paper)
## 3. Comparison of Methods
## 4. Evolution / Progress of Methods
(chronological / conceptual progression between the papers)
## 5. Conclusion

PAPERS:
{chr(10).join(paper_summaries)}
"""
    survey = llm_client.ask(prompt, system=system, temperature=0.3)

    # ایجاد پوشه در صورت عدم وجود
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUTS_DIR, "mini_survey.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(survey)
    return survey


# ---------------------------------------------------------------------------
# تابع اختیاری اول: find_research_gap
# ---------------------------------------------------------------------------
def find_research_gap(paper_ids: list = None) -> dict:
    papers = load_all_papers()
    if paper_ids:
        papers = [p for p in papers if p["paper_id"] in paper_ids]

    output = {}
    for p in papers:
        text = _get_section(p, ["Conclusion", "Future Work", "Limitations"])
        if not text:
            text = _get_section(p, ["Conclusion"])
        system = "You analyze a paper's conclusion/limitations/future-work text and answer in strict JSON."
        prompt = f"""
From the following conclusion/limitations/future-work text of the paper "{p['title']}",
extract:
1. limitations: known weaknesses/limitations stated by the authors
2. unsolved_problems: problems that remain unsolved
3. future_directions: suggested future research directions

Return strict JSON: {{"limitations": [...], "unsolved_problems": [...], "future_directions": [...]}}

TEXT:
{text}
"""
        parsed = llm_client.ask_json(prompt, system=system)
        output[p["paper_id"]] = {"title": p["title"], **parsed}
    return output


# ---------------------------------------------------------------------------
# تابع اختیاری دوم: compare_references
# ---------------------------------------------------------------------------
YEAR_PATTERN = re.compile(r"(19|20)\d{2}")


def compare_references(paper_ids: list = None) -> dict:
    """
    بخش References هر مقاله را با regex پردازش کرده و تعداد رفرنس به تفکیک
    سال انتشار را استخراج می‌کند (بدون نیاز به LLM چون کار کاملاً ساختاریافته است).
    """
    papers = load_all_papers()
    if paper_ids:
        papers = [p for p in papers if p["paper_id"] in paper_ids]

    year_counts = {}
    titles = {}
    for p in papers:
        ref_text = _get_section(p, ["References"])
        # تفکیک رفرنس‌ها بر اساس [n] یا خطوط جدید که با حرف یا عدد شروع می‌شوند
        entries = re.split(r"\[\d+\]|\n\s*(?=\d+\.|[A-Z][a-z]+)", ref_text)
        counts = Counter()
        for entry in entries:
            if not entry.strip():
                continue
            full_years = re.findall(r"(?:19|20)\d{2}", entry)
            if full_years:
                counts[full_years[-1]] += 1
        year_counts[p["paper_id"]] = dict(counts)
        titles[p["paper_id"]] = p["title"]

    all_years = sorted({y for counts in year_counts.values() for y in counts})
    pids = list(year_counts.keys())

    header = "| Year | " + " | ".join(titles[pid][:25] for pid in pids) + " |"
    sep = "|------|" + "------|" * len(pids)
    rows = [header, sep]
    for y in all_years:
        row = [y] + [str(year_counts[pid].get(y, 0)) for pid in pids]
        rows.append("| " + " | ".join(row) + " |")

    return {"table_markdown": "\n".join(rows), "raw": year_counts, "titles": titles}


# ---------------------------------------------------------------------------
# تابع اختیاری سوم: compare_limitations
# ---------------------------------------------------------------------------
def compare_limitations(paper_ids: list = None) -> str:
    papers = load_all_papers()
    if paper_ids:
        papers = [p for p in papers if p["paper_id"] in paper_ids]

    blocks = []
    for p in papers:
        text = _get_section(p, ["Limitations", "Conclusion", "Future Work"])
        blocks.append(f"### {p['title']}\n{text[:2000]}")

    system = "You compare the limitations of several computer-vision papers and answer in markdown."
    prompt = f"""
Compare the limitations / weaknesses of the following papers. Produce a markdown table with
columns: Paper | Main Limitations | Relative Weakness Compared to Others, followed by a short
paragraph summary.

{chr(10).join(blocks)}
"""
    return llm_client.ask(prompt, system=system, temperature=0.3)
