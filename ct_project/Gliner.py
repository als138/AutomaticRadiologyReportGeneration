"""
GLiNER Model Test Script
========================
این اسکریپت برای تست مدل GLiNER روی متن‌های فارسی و انگلیسی طراحی شده است.

نیازمندی‌ها:
- torch
- GLiNER
- numpy

برای نصب:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/urchade/GLiNER.git
"""

import torch
import numpy as np
# اگر GLiNER را دستی دانلود کردید، این خط را uncomment کنید:
# import sys
# sys.path.append('./gliner')
from gliner import GLiNER


def check_environment():
    """بررسی محیط و GPU"""
    print("=== بررسی محیط ===")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    else:
        print("Device: CPU")
    print(f"NumPy Version: {np.__version__}")
    print(f"PyTorch Version: {torch.__version__}")
    print()


def load_gliner_model():
    """بارگذاری مدل GLiNER"""
    print("=== بارگذاری مدل GLiNER ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on: {device}")
    
    try:
        model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1").to(device)
        print("✅ مدل GLiNER با موفقیت بارگذاری شد")
        return model, device
    except Exception as e:
        print(f"❌ خطا در بارگذاری مدل: {e}")
        return None, device


def test_english_text(model):
    """تست مدل روی متن انگلیسی"""
    print("=== تست متن انگلیسی ===")
    text = "Barack Obama was born in Hawaii. He was president of the United States."
    labels = ["person", "location", "organization"]
    
    print(f"Text: {text}")
    print(f"Labels: {labels}")
    print("Results:")
    
    entities = model.predict_entities(text, labels)
    for ent in entities:
        print(f"  {ent['text']} → {ent['label']} (score: {ent['score']:.2f})")
    print()


def test_persian_text(model):
    """تست مدل روی متن فارسی"""
    print("=== تست متن فارسی ===")
    text = "علی دایی در اردبیل به دنیا آمد و برای پرسپولیس بازی کرد."
    labels = ["شخص", "شهر", "تیم"]
    
    print(f"Text: {text}")
    print(f"Labels: {labels}")
    print("Results:")
    
    entities = model.predict_entities(text, labels)
    for ent in entities:
        print(f"  {ent['text']} → {ent['label']} (score: {ent['score']:.2f})")
    print()


def test_mixed_entities(model):
    """تست مدل با انواع مختلف موجودیت‌ها"""
    print("=== تست با انواع مختلف موجودیت‌ها ===")
    
    # تست انگلیسی پیشرفته
    text_en = "Apple Inc. was founded by Steve Jobs in Cupertino, California in 1976."
    labels_en = ["person", "organization", "location", "date"]
    
    print(f"English Text: {text_en}")
    print("English Results:")
    entities = model.predict_entities(text_en, labels_en)
    for ent in entities:
        print(f"  {ent['text']} → {ent['label']} (score: {ent['score']:.2f})")
    
    print()
    
    # تست فارسی پیشرفته
    text_fa = "حافظ شیرازی شاعر بزرگ ایرانی در شیراز زندگی می‌کرد و در قرن چهاردهم میلادی می‌زیست."
    labels_fa = ["شخص", "شهر", "کشور", "زمان"]
    
    print(f"Persian Text: {text_fa}")
    print("Persian Results:")
    entities = model.predict_entities(text_fa, labels_fa)
    for ent in entities:
        print(f"  {ent['text']} → {ent['label']} (score: {ent['score']:.2f})")
    print()


def main():
    """تابع اصلی برای اجرای تست‌ها"""
    print("🚀 شروع تست مدل GLiNER")
    print("=" * 50)
    
    # بررسی محیط
    check_environment()
    
    # بارگذاری مدل
    model, device = load_gliner_model()
    if model is None:
        print("❌ امکان ادامه تست وجود ندارد")
        return
    
    print("=" * 50)
    
    # اجرای تست‌ها
    test_english_text(model)
    test_persian_text(model)
    test_mixed_entities(model)
    
    print("✅ تمام تست‌ها با موفقیت انجام شد!")


if __name__ == "__main__":
    main()