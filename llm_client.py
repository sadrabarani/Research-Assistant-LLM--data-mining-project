# -*- coding: utf-8 -*-
"""
کلاینت ارتباط با مدل زبانی. بر اساس config.LLM_PROVIDER یکی از این دو مسیر
انتخاب می‌شود:
  - "ollama": مدل لوکال از طریق سرور Ollama (بدون هزینه، بدون اینترنت)
  - "groq":   API آنلاین و رایگان Groq (نیاز به GROQ_API_KEY، سرعت بسیار بالا)

بقیه پروژه (rag.py, functions.py, voice_input.py) فقط از توابع chat/ask/ask_json
استفاده می‌کنند و کاملاً مستقل از provider انتخابی هستند.
"""
import json
import requests

from config import (
    USE_GROQ_API,
    OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_TIMEOUT,
    GROQ_API_KEY, GROQ_MODEL, GROQ_TIMEOUT,
)


class LLMError(RuntimeError):
    pass


# ------------------------------------------------------------------
# Ollama backend
# ------------------------------------------------------------------
def _check_ollama_server():
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        r.raise_for_status()
    except Exception as e:
        raise LLMError(
            "اتصال به سرور Ollama برقرار نشد. مطمئن شوید Ollama نصب و در حال اجراست "
            f"(دستور: `ollama serve`) و مدل `{OLLAMA_MODEL}` را با "
            f"`ollama pull {OLLAMA_MODEL}` دانلود کرده‌اید. خطا: {e}"
        )


def _chat_ollama(messages, model, temperature, json_mode):
    _check_ollama_server()
    payload = {
        "model": model or OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if json_mode:
        payload["format"] = "json"
    try:
        resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=OLLAMA_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "").strip()
    except requests.exceptions.RequestException as e:
        raise LLMError(f"خطا در فراخوانی مدل Ollama: {e}")


def _chat_ollama_stream(messages, model, temperature):
    _check_ollama_server()
    payload = {
        "model": model or OLLAMA_MODEL,
        "messages": messages,
        "stream": True,
        "options": {"temperature": temperature},
    }
    try:
        with requests.post(f"{OLLAMA_HOST}/api/chat", json=payload,
                            timeout=OLLAMA_TIMEOUT, stream=True) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                chunk = data.get("message", {}).get("content", "")
                if chunk:
                    yield chunk
                if data.get("done"):
                    break
    except requests.exceptions.RequestException as e:
        raise LLMError(f"خطا در فراخوانی استریم Ollama: {e}")


def _stream_ollama(messages, model, temperature):
    _check_ollama_server()
    payload = {
        "model": model or OLLAMA_MODEL,
        "messages": messages,
        "stream": True,
        "options": {"temperature": temperature},
    }
    try:
        with requests.post(f"{OLLAMA_HOST}/api/chat", json=payload,
                            timeout=OLLAMA_TIMEOUT, stream=True) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                data = json.loads(line.decode("utf-8"))
                content = data.get("message", {}).get("content", "")
                if content:
                    yield content
                if data.get("done"):
                    break
    except requests.exceptions.RequestException as e:
        raise LLMError(f"خطا در فراخوانی مدل Ollama (stream): {e}")


# ------------------------------------------------------------------
# Groq backend (OpenAI-compatible REST API)
# ------------------------------------------------------------------
def _chat_groq(messages, model, temperature, json_mode):
    if not GROQ_API_KEY:
        raise LLMError(
            "GROQ_API_KEY تنظیم نشده است. توکن را از https://console.groq.com/keys بگیرید "
            "و به صورت متغیر محیطی ست کنید:\n"
            "  Windows (PowerShell): $env:GROQ_API_KEY=\"gsk_...\"\n"
            "  یا در config.py مقدار GROQ_API_KEY را مستقیم بنویسید."
        )
    payload = {
        "model": model or GROQ_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers, json=payload, timeout=GROQ_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except requests.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = resp.json().get("error", {}).get("message", "")
        except Exception:
            pass
        raise LLMError(f"خطای Groq API: {e} {detail}")
    except requests.exceptions.RequestException as e:
        raise LLMError(f"خطا در اتصال به Groq API: {e}")


def _chat_groq_stream(messages, model, temperature):
    if not GROQ_API_KEY:
        raise LLMError("GROQ_API_KEY تنظیم نشده است.")
    payload = {
        "model": model or GROQ_MODEL,
        "messages": messages,
        "temperature": temperature,
        "stream": True,
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        with requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers, json=payload, timeout=GROQ_TIMEOUT, stream=True,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8") if isinstance(line, bytes) else line
                if not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                delta = data.get("choices", [{}])[0].get("delta", {})
                chunk = delta.get("content", "")
                if chunk:
                    yield chunk
    except requests.exceptions.RequestException as e:
        raise LLMError(f"خطا در استریم Groq API: {e}")


def _stream_groq(messages, model, temperature):
    if not GROQ_API_KEY:
        raise LLMError("GROQ_API_KEY تنظیم نشده است.")
    payload = {
        "model": model or GROQ_MODEL,
        "messages": messages,
        "temperature": temperature,
        "stream": True,
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        with requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers, json=payload, timeout=GROQ_TIMEOUT, stream=True,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                delta = data.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content
    except requests.exceptions.RequestException as e:
        raise LLMError(f"خطا در اتصال به Groq API (stream): {e}")


# ------------------------------------------------------------------
# رابط عمومی و مستقل از provider
# ------------------------------------------------------------------
def chat(messages: list, model: str = None, temperature: float = 0.2,
         json_mode: bool = False) -> str:
    if USE_GROQ_API:
        return _chat_groq(messages, model, temperature, json_mode)
    return _chat_ollama(messages, model, temperature, json_mode)


def chat_stream(messages: list, model: str = None, temperature: float = 0.2):
    """نسخه استریم: به جای رشته کامل، یک generator از تکه‌متن‌ها برمی‌گرداند."""
    if USE_GROQ_API:
        yield from _chat_groq_stream(messages, model, temperature)
    else:
        yield from _chat_ollama_stream(messages, model, temperature)


def ask_stream(prompt: str, system: str = None, temperature: float = 0.2):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    yield from chat_stream(messages, temperature=temperature)


def ask(prompt: str, system: str = None, temperature: float = 0.2,
        json_mode: bool = False) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return chat(messages, temperature=temperature, json_mode=json_mode)


def stream_chat(messages: list, model: str = None, temperature: float = 0.2):
    """generator که تکه‌های متن پاسخ را همانطور که تولید می‌شوند yield می‌کند."""
    if USE_GROQ_API:
        yield from _stream_groq(messages, model, temperature)
    else:
        yield from _stream_ollama(messages, model, temperature)


def stream_ask(prompt: str, system: str = None, temperature: float = 0.2):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    yield from stream_chat(messages, temperature=temperature)


def ask_json(prompt: str, system: str = None) -> dict:
    """پاسخ مدل را به صورت JSON پارس می‌کند (با تلاش برای پاک‌سازی fenced code)."""
    json_system = (system or "") + "\nAlways respond with valid JSON only."
    raw = ask(prompt, system=json_system, json_mode=True)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                pass
        return {"_raw": raw, "_error": "قابل تبدیل به JSON نبود"}