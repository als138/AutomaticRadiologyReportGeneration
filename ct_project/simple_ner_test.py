"""
Simple NER Test with Transformers (بدون GLiNER)
===============================================
این فایل از transformers برای Named Entity Recognition استفاده می‌کند
"""

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

def test_transformers_ner():
    """تست NER با استفاده از transformers"""
    print("🚀 تست NER با Transformers")
    print("=" * 50)
    
    # بررسی دستگاه
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    try:
        # مدل NER انگلیسی
        ner_pipeline = pipeline(
            "ner", 
            model="dbmdz/bert-large-cased-finetuned-conll03-english",
            device=0 if device == "cuda" else -1,
            aggregation_strategy="simple"
        )
        
        # تست متن انگلیسی
        text_en = "Barack Obama was born in Hawaii. He was president of the United States."
        print(f"\nEnglish Text: {text_en}")
        
        entities = ner_pipeline(text_en)
        print("English Results:")
        for ent in entities:
            print(f"  {ent['word']} → {ent['entity_group']} (score: {ent['score']:.2f})")
        
        print("\n✅ تست موفقیت‌آمیز بود!")
        
    except Exception as e:
        print(f"❌ خطا: {e}")

def test_multilingual_ner():
    """تست NER چندزبانه"""
    print("\n" + "=" * 50)
    print("🌍 تست NER چندزبانه")
    print("=" * 50)
    
    try:
        # مدل چندزبانه
        ner_pipeline = pipeline(
            "ner",
            model="Davlan/distilbert-base-multilingual-cased-ner-hrl",
            device=0 if torch.cuda.is_available() else -1,
            aggregation_strategy="simple"
        )
        
        # تست متن فارسی
        text_fa = "علی دیابت دارد"
        print(f"Persian Text: {text_fa}")
        
        entities = ner_pipeline(text_fa)
        print("Persian Results:")
        for ent in entities:
            print(f"  {ent['word']} → {ent['entity_group']} (score: {ent['score']:.2f})")
            
        print("\n✅ تست چندزبانه موفقیت‌آمیز بود!")
        
    except Exception as e:
        print(f"❌ خطا در تست چندزبانه: {e}")

if __name__ == "__main__":
    test_transformers_ner()
    test_multilingual_ner()
