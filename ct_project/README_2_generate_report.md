# ✅ CT Scan Report Generation Pipeline (`2_generate_report.py`)

---

## 📖 Overview

This script implements a multi-stage, multi-lingual, RAG-based (Retrieval-Augmented Generation) pipeline for generating structured CT scan reports from medical images. It leverages:

- 🖼️ **LLaVA**: Vision-Language Model for extracting findings from CT images.
- 🩺 **MedGemma**: Large Language Model for medical report generation (English & Persian).
- 📚 **LangChain + FAISS**: For retrieval of relevant context from a knowledge base of prior reports (RAG).
- 📊 **Evaluation Metrics**: Automated comparison of RAG vs. no-RAG outputs, including Jaccard similarity and medical term coverage.

The pipeline produces and saves four types of reports (English/Persian, with/without RAG) and evaluates the impact of retrieval augmentation.

---

## ✨ Features

- 🖼️ **Image-to-Text**: Extracts preliminary findings from CT images using LLaVA.
- 📝 **Report Generation**: Produces structured reports in both English and Persian, with and without RAG.
- 🔍 **Retrieval-Augmented Generation**: Incorporates relevant context from a FAISS-based vector store of prior reports.
- 📊 **Evaluation**: Computes Jaccard similarity and medical term coverage to assess RAG effectiveness.
- 🧹 **Clean Output**: Removes system prompts, duplicates, and extraneous text for clarity.
- 🌐 **Multi-lingual**: Persian and English support, with Persian section titles.
- 💾 **All results saved**: Outputs and evaluation are saved to a timestamped file in `final-reports/`.

---

## ⚙️ Requirements

- Python 3.8+
- CUDA-enabled GPU (recommended for LLaVA/MedGemma)
- All dependencies listed in `requirements_gliner.txt`
- Pre-built vector store in `vector_store/` (see `1_prepare_knowledge_base.py`)
- Test images in `test_images/`

---

## 🚀 Usage

1. **Prepare the Knowledge Base**  
   Run `1_prepare_knowledge_base.py` first to build the FAISS vector store from your report corpus.

2. **Run the Report Generation Pipeline**  
   ```powershell
   python 2_generate_report.py
   ```

3. **Outputs**  
   - Four reports are generated and printed:
     1. English report WITHOUT RAG
     2. Persian report WITHOUT RAG
     3. English report WITH RAG
     4. Persian report WITH RAG
   - Evaluation metrics are printed and appended.
   - All results are saved to a timestamped file in `final-reports/`.

---

## 🛠️ Pipeline Stages

1. **Model Loading**
   - Loads LLaVA (vision), MedGemma (language), and BioBERT (embeddings).

2. **Image Analysis**
   - Extracts concise findings from the CT image.

3. **Report Generation**
   - Generates English and Persian reports, both with and without RAG context.

4. **Retrieval**
   - For RAG, retrieves top-5 similar reports from the vector store using LangChain/FAISS.

5. **Evaluation**
   - Calculates Jaccard similarity between RAG and no-RAG reports.
   - Counts medical term occurrences in each report.

6. **Saving Results**
   - All outputs and evaluation metrics are saved to a file in `final-reports/`.

---

## 🧩 Customization

- **Change the test image**: Edit `TEST_IMAGE_PATH` at the top of the script.
- **Modify prompt templates**: Edit the prompt strings in the script for different report structures or languages.
- **Add more evaluation metrics**: Extend the evaluation section as needed.

---

## 📦 Example Output

```
======================================================================
🔍 1. English Report WITHOUT RAG
======================================================================
FINDINGS:
...
IMPRESSION:
...
------------------------------------------------------------------------------------
...
RAG EVALUATION RESULTS
======================================================================
English Jaccard Similarity (NoRAG vs RAG): 0.72
Persian Jaccard Similarity (NoRAG vs RAG): 0.68
Medical Terms in English NoRAG: 5
Medical Terms in English RAG: 7
...
```

---

## 📝 Notes

- All code comments and print statements are in English, except for Persian section titles.
- Ensure the vector store is built and available before running this script.
- For best results, use a GPU and ensure all models are downloaded and cached.

---

## 📚 Citation

If you use this pipeline in your research, please cite the relevant model papers (LLaVA, MedGemma, BioBERT, LangChain).

---

## 📬 Contact

For questions or issues, please contact the project maintainer.
