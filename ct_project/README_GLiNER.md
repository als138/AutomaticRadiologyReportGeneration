# GLiNER Model Test

این اسکریپت برای تست مدل GLiNER (Generalist and Lightweight Named Entity Recognition) روی متن‌های فارسی و انگلیسی طراحی شده است.

## نصب

### روش 1: استفاده از requirements.txt
```bash
pip install -r requirements_gliner.txt
```

### روش 2: نصب دستی
```bash
# نصب PyTorch (برای GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# نصب GLiNER
pip install git+https://github.com/urchade/GLiNER.git

# نصب NumPy
pip install numpy
```

## اجرا

```bash
python Gliner.py
```

## ویژگی‌ها

- ✅ بررسی محیط و GPU
- ✅ بارگذاری مدل GLiNER Multi v2.1
- ✅ تست روی متن انگلیسی
- ✅ تست روی متن فارسی
- ✅ تست با انواع مختلف موجودیت‌ها
- ✅ مدیریت خطا

## نمونه خروجی

```
🚀 شروع تست مدل GLiNER
==================================================
=== بررسی محیط ===
CUDA Available: True
GPU Device: NVIDIA GeForce RTX 3080
NumPy Version: 1.24.3
PyTorch Version: 2.0.1

=== بارگذاری مدل GLiNER ===
Loading model on: cuda
✅ مدل GLiNER با موفقیت بارگذاری شد
==================================================
=== تست متن انگلیسی ===
Text: Barack Obama was born in Hawaii. He was president of the United States.
Labels: ['person', 'location', 'organization']
Results:
  Barack Obama → person (score: 0.99)
  Hawaii → location (score: 0.95)
  United States → location (score: 0.92)

=== تست متن فارسی ===
Text: علی دایی در اردبیل به دنیا آمد و برای پرسپولیس بازی کرد.
Labels: ['شخص', 'شهر', 'تیم']
Results:
  علی دایی → شخص (score: 0.87)
  اردبیل → شهر (score: 0.82)
  پرسپولیس → تیم (score: 0.85)
```

## توضیح کدها

### check_environment()
بررسی محیط، GPU، و ورژن کتابخانه‌ها

### load_gliner_model()
بارگذاری مدل GLiNER با مدیریت خطا

### test_english_text()
تست مدل روی متن انگلیسی با شناسایی شخص، مکان، و سازمان

### test_persian_text()
تست مدل روی متن فارسی با برچسب‌های فارسی

### test_mixed_entities()
تست‌های پیشرفته‌تر با انواع مختلف موجودیت‌ها

## نکات مهم

- مدل GLiNER چندزبانه است و از فارسی پشتیبانی می‌کند
- برای عملکرد بهتر از GPU استفاده می‌شود
- برچسب‌ها (labels) باید مطابق با نوع موجودیت‌های مورد نظر تعریف شوند
