import os
import torch
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# --- تنظیمات ---
REPORTS_PATH = "data_reports"
VECTORSTORE_PATH = "vector_store"
EMBEDDING_MODEL_NAME = "pritamdeka/S-BioBert-snli-multinli-stsb"

def create_knowledge_base():
    """
    این تابع گزارش‌های متنی را خوانده، آن‌ها را به قطعات کوچک تقسیم کرده،
    به بردارهای امبدینگ تبدیل می‌کند و در یک پایگاه داده FAISS ذخیره می‌کند.
    """
    if os.path.exists(VECTORSTORE_PATH) and os.listdir(VECTORSTORE_PATH):
        print(f"پایگاه دانش در مسیر '{VECTORSTORE_PATH}' از قبل موجود است. از ساخت مجدد صرف نظر شد.")
        return

    print("شروع ساخت پایگاه دانش...")

    # 1. بارگذاری تمام گزارش‌ها از پوشه
    loader = DirectoryLoader(REPORTS_PATH, glob="**/*.txt", show_progress=True, loader_kwargs={'encoding': 'utf-8'})
    documents = loader.load()
    if not documents:
        print("خطا: هیچ گزارشی در پوشه data_reports یافت نشد. لطفاً فایل‌های .txt خود را آنجا قرار دهید.")
        return

    # 2. خرد کردن اسناد به قطعات کوچک‌تر
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    print(f"تعداد {len(documents)} گزارش به {len(docs)} قطعه تقسیم شد.")

    # 3. انتخاب و بارگذاری مدل Embedding
    # این مدل برای متون پزشکی بهینه شده است
    print(f"درحال بارگذاری مدل امبدینگ: {EMBEDDING_MODEL_NAME}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={'device': device})

    # 4. ساخت پایگاه داده FAISS و ذخیره‌سازی
    print("درحال ساخت و ذخیره‌سازی پایگاه داده برداری FAISS...")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)

    print(f"✅ پایگاه دانش با موفقیت ساخته و در '{VECTORSTORE_PATH}' ذخیره شد.")

if __name__ == "__main__":
    create_knowledge_base()