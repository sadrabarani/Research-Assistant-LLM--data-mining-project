"""
تنظیمات کلی پروژه دستیار پژوهشی هوشمند
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PAPERS_DIR = os.path.join(BASE_DIR, "data", "papers")
KB_DIR = os.path.join(BASE_DIR, "data", "kb")          # متادیتای استخراج‌شده از مقالات (JSON)
OUTPUTS_DIR = os.path.join(BASE_DIR, "data", "outputs")  # نمودار/جدول/گزارش خروجی
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

# ---------------- انتخاب Provider برای LLM ----------------
# اگر True باشد از API آنلاین Groq استفاده می‌شود، اگر False باشد از Ollama لوکال.
# می‌توانید همینجا مستقیم True/False کنید (نیازی به تنظیم متغیر محیطی نیست).
# USE_GROQ_API = True
USE_GROQ_API = os.environ.get("USE_GROQ_API", "False").lower() in ("1", "true", "yes")

# ---------------- Ollama (LLM لوکال) ----------------
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
# هر مدلی که با ollama pull گرفته باشید قابل استفاده است (llama3.1, qwen2.5, mistral, ...)
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_TIMEOUT = 300

# ---------------- Groq (LLM ابری، رایگان با rate-limit) ----------------
GROQ_API_KEY = "YOUR_API_KEY_HERE"
# مدل قدرتمند لاما که بالاترین دقت را در تحلیل مقالات دارد
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_TIMEOUT = 120

# ---------------- Embedding (لوکال) ----------------
# مدل چندزبانه چون مقالات انگلیسی هستند ولی سوالات ممکن است فارسی باشند
EMBEDDING_MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2"
)

# ---------------- Whisper (صوت) ----------------
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "base")  # tiny/base/small/medium

# ---------------- Chunking ----------------
CHUNK_SIZE_WORDS = 220      # اندازه هر chunk بر حسب تعداد کلمه
CHUNK_OVERLAP_WORDS = 40    # همپوشانی بین chunk های متوالی

# ---------------- Retrieval ----------------
TOP_K = 6
# اگر بیشترین شباهت بازیابی‌شده از این آستانه کمتر باشد یعنی پاسخ در مقالات نیست
SIMILARITY_DISTANCE_THRESHOLD = 0.85  # فاصله کسینوسی (کمتر = شبیه‌تر) - وابسته به chromadb

SECTION_NAMES = [
    "Abstract", "Introduction", "Related Work", "Background",
    "Method", "Methodology", "Approach", "Experiments", "Experiment",
    "Results", "Evaluation", "Discussion", "Conclusion", "Conclusions",
    "Limitations", "Future Work", "References"
]

os.makedirs(KB_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(PAPERS_DIR, exist_ok=True)