# ✅ Knowledge Base Preparation (`1_prepare_knowledge_base.py`)

---

## 📖 Overview

This script builds a searchable knowledge base of medical CT scan reports for use in Retrieval-Augmented Generation (RAG) pipelines. It processes a directory of text reports, splits them into chunks, embeds them using a medical embedding model, and stores them in a FAISS vector database for fast similarity search.

---

## ✨ Features

- 📂 **Corpus Ingestion**: Loads all `.txt` reports from a specified directory.
- ✂️ **Text Splitting**: Splits long reports into manageable chunks for better retrieval.
- 🧬 **Medical Embeddings**: Uses BioBERT or similar models for domain-specific embeddings.
- 🗂️ **FAISS Vector Store**: Stores all embeddings for efficient similarity search.
- 🔗 **Ready for RAG**: Output is directly usable by `2_generate_report.py` for context retrieval.

---

## ⚙️ Requirements

- Python 3.8+
- All dependencies listed in `requirements_gliner.txt`
- Directory of `.txt` medical reports (default: `data_reports/`)

---

## 🚀 Usage

1. **Place your medical reports**  
   Ensure all your `.txt` reports are in the `data_reports/` directory.

2. **Run the script**  
   ```powershell
   python 1_prepare_knowledge_base.py
   ```

3. **Output**  
   - A FAISS vector store is created in the `vector_store/` directory.
   - This store is used by the report generation pipeline for retrieval.

---

## 🛠️ Pipeline Steps

1. **Load Reports**
   - Reads all `.txt` files from the reports directory.

2. **Split Text**
   - Splits each report into smaller chunks (e.g., by paragraph or fixed length).

3. **Embed Chunks**
   - Uses a medical embedding model (e.g., BioBERT) to convert text chunks into vectors.

4. **Build Vector Store**
   - Stores all vectors in a FAISS index for fast similarity search.

5. **Save Vector Store**
   - Saves the FAISS index and metadata to `vector_store/`.

---

## 🧩 Customization

- **Change report directory**: Edit the path to your reports in the script.
- **Change embedding model**: Modify the embedding model name as needed.
- **Adjust chunk size**: Change the splitting logic for different chunk sizes.

---

## 📝 Notes

- This script must be run before `2_generate_report.py`.
- The quality of retrieval depends on the diversity and quality of your report corpus.
- For large corpora, ensure sufficient disk space and memory.

---

## 📬 Contact

For questions or issues, please contact the project maintainer.
