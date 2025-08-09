# Automatic Radiology Report Generation using RAG and LLMs

A full end-to-end pipeline for automatic generation of CT scan reports using Vision–Language Models (VLMs) and Large Language Models (LLMs), with a RAG (Retrieval‑Augmented Generation) architecture to boost accuracy via a local knowledge base.

---

## 🏗️ Project Architecture

The system consists of two main stages:

1. **Image Analysis**  
   A VLM (e.g. LLaVA) processes the CT scan image and outputs preliminary findings in text.

2. **RAG‑Enhanced Report Generation**  
   Preliminary findings are augmented with similar reports retrieved from a local FAISS knowledge base, and fed into an LLM (e.g. Gemma) to generate a structured final report.

**Pipeline Flow:**

```
[CT Image] → [LLaVA Vision Model] → [Preliminary Findings (Text)]
               ↓                                ↓
      [RAG Retriever: FAISS] → → → → → → → → [Gemma LLM] → [Final Report]
```

---

## 🔍 Features

- **End‑to‑End Pipeline**: From image input to structured text output  
- **Local Execution**: Full operation locally to preserve privacy  
- **RAG‑Based**: Retrieves similar historical reports to increase accuracy and coherence  
- **Highly Customizable**: Swap LLM, VLM, or embedding models to fit your needs

---

## ⚙️ Technology Stack

- **LLM**: `google/gemma-7b-it` (or `google/gemma-2b`)  
- **VLM**: `llava‑hf/llava‑1.5‑7b‑hf`  
- **Embedding Model**: `pritamdeka/S‑BioBert‑snli‑multinli‑stsb`  
- **Vector DB**: FAISS (Facebook AI Similarity Search)  
- **Frameworks**: LangChain, Transformers, PyTorch

---

## 🚀 Setup and Installation

### 1. Prerequisites

- Python 3.10+  
- Git  
- NVIDIA GPU (min. 16 GB VRAM; 24 GB recommended)

### 2. Clone the Repository

```bash
git clone <YOUR‑REPOSITORY‑URL>
cd <YOUR‑PROJECT‑DIRECTORY>
```

### 3. Directory Structure

Ensure your project root matches the following layout:

```
/
|-- data_reports/       # All .txt medical report files
|-- test_images/        # Sample CT scan images
|-- vector_store/       # Directory where FAISS stores vectors
|-- 1_prepare_knowledge_base.py
|-- 2_generate_report.py
└-- README.md
```

### 4. Install Dependencies

Use a virtual environment:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

**requirements.txt** should include:

```
torch
torchvision
torchaudio
langchain
langchain-community
transformers
sentence-transformers
faiss-cpu    # or faiss-gpu if on Linux
bitsandbytes
accelerate
pypdf
unstructured
Pillow
```

---

## 🎯 How to Use

### 1. Prepare Data  
Place anonymized `.txt` medical report files into `data_reports/`.

### 2. Hugging Face Authentication  

```bash
huggingface-cli login
```

Accept terms for:

- `google/gemma-7b-it`  
- `llava‑hf/llava‑1.5‑7b‑hf`

### 3. Build the Knowledge Base  

```bash
python 1_prepare_knowledge_base.py
```

### 4. Generate a Report  

- Place a CT scan sample (e.g. `sample.jpg`) into `test_images/`  
- Update `TEST_IMAGE_PATH` in `2_generate_report.py`  
- Run:

```bash
python 2_generate_report.py
```

The final report will appear in the console.

---

## ⚙️ Configuration

Update the following parameters at the top of `2_generate_report.py`:

- `LLM_MODEL_ID`  
- `VLM_MODEL_ID`  
- `TEST_IMAGE_PATH`

---

## ⚠️ Disclaimer

This project is strictly a research Proof of Concept.  
**The generated reports MUST be reviewed and verified by a qualified human radiologist.**  
Do not use outputs for clinical diagnosis or medical decisions.

---

## 📝 License

This project is licensed under the **MIT License**.  
See the `LICENSE` file in the repository for more details.

---

## ℹ️ Further Customization

- Add **Usage Examples** (include sample inputs and expected outputs)  
- Include **Badge Icons** (e.g. CI status, test coverage, version) at the top  
- Add sections like **Contribution Guidelines** or **Code of Conduct**

