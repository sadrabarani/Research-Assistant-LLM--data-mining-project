# -*- coding: utf-8 -*-
"""
کلاینت ارتباط با مدل زبانی لوکال از طریق Ollama.
پیش‌نیاز: نصب Ollama (https://ollama.com) و اجرای `ollama pull <model>`.
"""
import json
import requests

from config import OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_TIMEOUT


class OllamaError(RuntimeError):
    pass


def _check_server():
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        r.raise_for_status()
    except Exception as e:
        raise OllamaError(
            "اتصال به سرور Ollama برقرار نشد. مطمئن شوید Ollama نصب و در حال اجراست "
            f"(دستور: `ollama serve`) و مدل `{OLLAMA_MODEL}` را با "
            f"`ollama pull {OLLAMA_MODEL}` دانلود کرده‌اید. خطا: {e}"
        )


def chat(messages: list, model: str = None, temperature: float = 0.2,
         json_mode: bool = False) -> str:
    """
    messages: [{"role": "system"/"user"/"assistant", "content": "..."}]
    خروجی: متن پاسخ مدل
    """
    model = model or OLLAMA_MODEL
    _check_server()

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if json_mode:
        payload["format"] = "json"

    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/chat", json=payload, timeout=OLLAMA_TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "").strip()
    except requests.exceptions.RequestException as e:
        raise OllamaError(f"خطا در فراخوانی مدل Ollama: {e}")


def ask(prompt: str, system: str = None, temperature: float = 0.2,
        json_mode: bool = False) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return chat(messages, temperature=temperature, json_mode=json_mode)


def ask_json(prompt: str, system: str = None) -> dict:
    """پاسخ مدل را به صورت JSON پارس می‌کند (با تلاش برای پاک‌سازی fenced code)."""
    raw = ask(prompt, system=system, json_mode=True)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # تلاش برای پیدا کردن اولین { ... } معتبر در متن
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                pass
        return {"_raw": raw, "_error": "قابل تبدیل به JSON نبود"}
