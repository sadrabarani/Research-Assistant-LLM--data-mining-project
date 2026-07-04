# دستیار پژوهشی هوشمند مبتنی بر LLM

پیاده‌سازی پروژه دوم درس مبانی داده‌کاوی — یک دستیار پژوهشی مبتنی بر RAG که
۴ مقاله حوزه Object Detection (RT-DETR, RT-DETRv3, YOLOv10 + یک مقاله چهارم
که خودتان اضافه می‌کنید) را تحلیل، مقایسه و جمع‌بندی می‌کند.

**پشته فناوری (کاملاً لوکال، بدون هزینه API):**
| بخش | ابزار |
|---|---|
| استخراج متن PDF | PyMuPDF (fitz) با الگوریتم تشخیص دو-ستونی سفارشی |
| Chunking | آگاه از بخش‌بندی مقاله + پنجره لغزان با هم‌پوشانی |
| Embedding | `sentence-transformers` (لوکال، رایگان) |
| Vector DB | ChromaDB |
| LLM (تولید پاسخ / function calling) | Ollama (هر مدل لوکال دلخواه) |
| صوت | OpenAI Whisper (لوکال، متن‌باز) + پالایش با LLM |
| رابط کاربری | CLI + Gradio |

## نصب

```bash
# 1) نصب Ollama و دانلود یک مدل
# https://ollama.com/download
ollama pull llama3.1        # یا هر مدل دیگر (qwen2.5, mistral, ...)
ollama serve                 # سرور Ollama را روشن نگه دارید

# 2) نصب پیش‌نیازهای پایتون
pip install -r requirements.txt
# اگر ffmpeg نصب نیست (لازم برای Whisper):
# Ubuntu/Debian: sudo apt install ffmpeg
# macOS:         brew install ffmpeg

# 3) مقالات PDF خود را داخل پوشه data/papers/ قرار دهید (حداقل ۴ مقاله)
```

در صورت نیاز می‌توانید مدل و اندازه Whisper را در `config.py` تغییر دهید:
```python
OLLAMA_MODEL = "llama3.1"          # نام مدل نصب‌شده در Ollama
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
WHISPER_MODEL_SIZE = "small"       # tiny/base/small/medium
```

## اجرا

### رابط ترمینالی (CLI - حداقل الزامی)
```bash
python cli_app.py
```
اولین بار به صورت خودکار پایگاه دانش را می‌سازد. برای پرسش صوتی:
```
Ask your question:
> voice:/path/to/question.wav
```

### رابط وب پیشرفته (Gradio - امتیازی)
```bash
python gradio_app.py
```
شامل تب‌های: گفتگو (متن/صوت)، مقایسه نتایج + نمودار، تولید Mini-Survey،
شکاف‌های پژوهشی، مقایسه رفرنس‌ها، مقایسه محدودیت‌ها.

### ساخت مجدد دستی پایگاه دانش
```bash
python knowledge_base.py
```

## توضیح استراتژی Chunking

هر مقاله ابتدا با تشخیص عناوین استاندارد (Abstract, Introduction, Method,
Results, Conclusion, References, ...) به بخش تقسیم می‌شود، سپس هر بخش با
پنجره لغزان ۲۲۰ کلمه‌ای و هم‌پوشانی ۴۰ کلمه به chunk تبدیل می‌شود. این کار
باعث می‌شود:
1. یک chunk هرگز مرز دو بخش معنایی متفاوت (مثلاً پایان Introduction و شروع
   Method) را با هم قاطی نکند → embedding منسجم‌تر → بازیابی دقیق‌تر.
2. توابعی مثل `extract_experimental_results` بتوانند مستقیم و بدون نیاز به
   جستجوی معنایی، سراغ بخش Results/Experiments مقاله بروند.
3. هم‌پوشانی ۴۰ کلمه‌ای از قطع‌شدن جمله/عدد مهم درست روی مرز دو chunk
   جلوگیری می‌کند.

## راه‌حل چالش PDF دو ستونی

بسیاری از مقالات علمی (هر ۳ مقاله این پروژه) دوستونی چاپ شده‌اند. کتابخانه‌های
ساده متن ستون چپ و راست را قاطی می‌کنند. در `pdf_extractor.py` با استفاده از
`page.get_text("blocks")` در PyMuPDF، بلوک‌های متنی هر صفحه بر اساس مختصات x
به ستون چپ/راست دسته‌بندی و هرکدام جداگانه بر اساس محور y مرتب می‌شوند، سپس
ستون چپ به طور کامل و بعد ستون راست خوانده می‌شود (به همراه تشخیص بلوک‌های
سراسری مثل تیتر مقاله).

## رفتار در نبود اطلاعات مرتبط (بخش هفتم)

اگر نزدیک‌ترین نتیجه بازیابی‌شده از آستانه شباهت (`SIMILARITY_DISTANCE_THRESHOLD`
در `config.py`) دورتر باشد، سیستم دقیقاً طبق فرمت خواسته‌شده پاسخ می‌دهد:

```
پاسخ شما در مقالات انتخاب‌شده یافت نشد.
اما با استفاده از مدل زبانی، پاسخ عمومی به شرح زیر است:
...
```

## ساختار پروژه
```
config.py            تنظیمات کلی
pdf_extractor.py      استخراج متن PDF (حل مشکل دو ستونی) + بخش‌بندی
chunking.py           استراتژی chunking
vectorstore.py        رابط ChromaDB + embedding لوکال
llm_client.py         کلاینت Ollama
knowledge_base.py     خط لوله ساخت پایگاه دانش
functions.py          ۶ تابع function-calling (۳ الزامی + ۳ اختیاری)
rag.py                مسیریابی function-calling + پاسخ‌گویی RAG + fallback
voice_input.py        Whisper + پالایش با LLM
cli_app.py            رابط CLI
gradio_app.py         رابط Gradio
data/papers/          فایل‌های PDF مقالات (شما باید مقاله چهارم را اضافه کنید)
data/kb/              JSON بخش‌بندی‌شده هر مقاله
data/outputs/         خروجی‌های تولیدشده (نمودار، survey)
chroma_db/            دیتابیس برداری پایدار
```

## محدودیت‌های شناخته‌شده
- کیفیت خروجی به مدل لوکال انتخابی در Ollama وابسته است؛ مدل‌های کوچک‌تر ممکن
  است در function calling/JSON خروجی کمتر دقیق باشند.
- تشخیص عنوان مقاله از PDF یک heuristic ساده است و در چیدمان‌های غیرمعمول
  ممکن است دقیق نباشد.
- آستانه تشخیص «عدم وجود پاسخ در مقالات» (`SIMILARITY_DISTANCE_THRESHOLD`) بر
  اساس فاصله کسینوسی embedding تنظیم شده و ممکن است نیاز به تنظیم دستی
  بر اساس مدل embedding انتخابی داشته باشد.
