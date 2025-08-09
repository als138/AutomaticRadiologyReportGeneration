import torch
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)
from PIL import Image
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.llms import HuggingFacePipeline
import os
from datetime import datetime

# --- Settings ---
VECTORSTORE_PATH = "vector_store"
EMBEDDING_MODEL_NAME = "pritamdeka/S-BioBert-snli-multinli-stsb"
VLM_MODEL_ID = "llava-hf/llava-1.5-7b-hf"          # Vision model
LLM_MODEL_ID = "google/medgemma-4b-it"            # Language model (MedGemma 4B-IT)
# TEST_IMAGE_PATH = "test_images/images_medium_rg.2020200159.fig2.gif"  # Previous GIF file
TEST_IMAGE_PATH = "test_images/images_medium_rg.2020200159.fig2.gif"  # Current JPEG file

# --- 1. Model loading ---
def load_models():
    print("Starting to load models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Vision model (LLaVA)
    vlm_processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
    vlm_model = LlavaForConditionalGeneration.from_pretrained(
        VLM_MODEL_ID,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)
    print("✅ Vision model (LLaVA) loaded.")

    # Language model (MedGemma 4B-IT)
    tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL_ID,
        use_auth_token=True
    )
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_auth_token=True
    )
    pipe = pipeline(
        "text-generation",
        model=llm_model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        repetition_penalty=1.1
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    print("✅ Language model (MedGemma 4B-IT) loaded.")

    # Embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': device}
    )
    print("✅ Embedding model loaded.")

    return (vlm_processor, vlm_model, device), llm, embeddings

# --- 2. Helper functions ---
def clean_report_output(text):
    """Clean report output from extra and duplicate lines"""
    if not text:
        return ""
    
    # Remove system messages and prompts
    lines = text.split('\n')
    cleaned_lines = []
    
    skip_patterns = [
        'You are a professional radiologist',
        'IMAGE FINDINGS:',
        'CONTEXT FROM EXISTING REPORTS:',
        'FINAL REPORT',
        'Please provide',
        'USER:',
        'As a radiologist, describe',
        'ER:',
        '$\\boxed{',
        'Final Answer:',
        '---',
        'The final answer is'
    ]
    
    for line in lines:
        line = line.strip()
        if line and not any(pattern in line for pattern in skip_patterns):
            # Prevent duplicate lines
            if not cleaned_lines or line != cleaned_lines[-1]:
                cleaned_lines.append(line)
    
    # Remove extra empty lines
    result = []
    for i, line in enumerate(cleaned_lines):
        if line or (i > 0 and cleaned_lines[i-1]):
            result.append(line)
    
    return '\n'.join(result).strip()

def get_findings_from_image(image_path, vlm_components):
    vlm_processor, vlm_model, device = vlm_components

    print(f"Analyzing image: {image_path}")
    try:
        raw_image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Image not found at '{image_path}'.")
        return None

    vlm_prompt = (
        "USER: <image>\n"
        "As a radiologist, describe the key findings in this CT scan. "
        "Be factual and concise. Provide only the findings, not the instruction."
    )

    # Use named parameters and move inputs to device
    inputs = vlm_processor(
        text=vlm_prompt,
        images=raw_image,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    output = vlm_model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False
    )
    findings = vlm_processor.decode(
        output[0][2:],  # Remove start tokens
        skip_special_tokens=True
    )

    print(f"Initial findings from image: {findings}")
    return findings

# --- 3. Main execution ---
def main():
    vlm_components, llm, embeddings = load_models()

    print(f"Loading knowledge base from: {VECTORSTORE_PATH}")
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # --- Prompt templates ---
    
    # 1. English report WITHOUT RAG
    template_english_no_rag = """
You are a professional radiologist. Write a structured CT scan report in English based ONLY on the image findings.

IMAGE FINDINGS:
{preliminary_findings}

Please provide a clean, structured report with FINDINGS and IMPRESSION sections:

FINDINGS:

IMPRESSION:
"""
    
    # 2. Persian report WITHOUT RAG
    template_persian_no_rag = """
You are a professional radiologist. Write a structured CT scan report in Persian based ONLY on the image findings.

IMAGE FINDINGS:
{preliminary_findings}

Please provide a clean, structured report in Persian with یافته‌ها and نتیجه‌گیری sections:

یافته‌ها:

نتیجه‌گیری:
"""
    
    # 3. English report WITH RAG
    template_english_rag = """
You are a professional radiologist. Write a structured CT scan report in English using the image findings and context from existing reports.

IMAGE FINDINGS:
{preliminary_findings}

CONTEXT FROM EXISTING REPORTS:
{context}

Please provide a clean, structured report with FINDINGS and IMPRESSION sections:

FINDINGS:

IMPRESSION:
"""
    
    # 4. Persian report WITH RAG
    template_persian_rag = """
You are a professional radiologist. Write a structured CT scan report in Persian using the image findings and context from existing reports.

IMAGE FINDINGS:
{preliminary_findings}

CONTEXT FROM EXISTING REPORTS:
{context}

Please provide a clean, structured report in Persian with یافته‌ها and نتیجه‌گیری sections:

یافته‌ها:

نتیجه‌گیری:
"""

    # Create prompt objects
    prompt_en_no_rag = PromptTemplate(template=template_english_no_rag, input_variables=["preliminary_findings"])
    prompt_fa_no_rag = PromptTemplate(template=template_persian_no_rag, input_variables=["preliminary_findings"])
    prompt_en_rag = PromptTemplate(template=template_english_rag, input_variables=["preliminary_findings", "context"])
    prompt_fa_rag = PromptTemplate(template=template_persian_rag, input_variables=["preliminary_findings", "context"])

    # Create chains
    chain_en_no_rag = (
        {"preliminary_findings": RunnablePassthrough()} | prompt_en_no_rag | llm | StrOutputParser()
    )
    
    chain_fa_no_rag = (
        {"preliminary_findings": RunnablePassthrough()} | prompt_fa_no_rag | llm | StrOutputParser()
    )
    
    chain_en_rag = (
        {"context": retriever, "preliminary_findings": RunnablePassthrough()} | prompt_en_rag | llm | StrOutputParser()
    )
    
    chain_fa_rag = (
        {"context": retriever, "preliminary_findings": RunnablePassthrough()} | prompt_fa_rag | llm | StrOutputParser()
    )

    preliminary_findings = get_findings_from_image(TEST_IMAGE_PATH, vlm_components)
    if preliminary_findings:
        # Clean initial findings
        preliminary_findings = clean_report_output(preliminary_findings)
        
        all_reports = ""
        
        # 1. English report WITHOUT RAG
        print("\n" + "="*70)
        print("🔍 1. English Report WITHOUT RAG")
        print("="*70)
        report_en_no_rag = chain_en_no_rag.invoke(preliminary_findings)
        report_en_no_rag = clean_report_output(report_en_no_rag)
        print(report_en_no_rag)
        print("------------------------------------------------------------------------------------")
        all_reports += "1. English Report WITHOUT RAG\n" + "="*70 + "\n" + report_en_no_rag + "\n" + "-"*84 + "\n\n"
        
        # 2. گزارش فارسی بدون RAG
        print("\n" + "="*70)
        print("🔍 2. گزارش فارسی بدون RAG")
        print("="*70)
        report_fa_no_rag = chain_fa_no_rag.invoke(preliminary_findings)
        report_fa_no_rag = clean_report_output(report_fa_no_rag)
        print(report_fa_no_rag)
        print("------------------------------------------------------------------------------------")
        all_reports += "2. گزارش فارسی بدون RAG\n" + "="*70 + "\n" + report_fa_no_rag + "\n" + "-"*84 + "\n\n"
        
        # 3. Announce using RAG
        print("\n" + "="*70)
        print("🚀 Using RAG for generate better reports")
        print("="*70)
        
        # 4. English report WITH RAG
        print("\n" + "="*70)
        print("🔍 3. English Report WITH RAG")
        print("="*70)
        report_en_rag = chain_en_rag.invoke(preliminary_findings)
        report_en_rag = clean_report_output(report_en_rag)
        print(report_en_rag)
        print("------------------------------------------------------------------------------------")
        all_reports += "3. English Report WITH RAG\n" + "="*70 + "\n" + report_en_rag + "\n" + "-"*84 + "\n\n"
        
        # 5. گزارش فارسی با RAG
        print("\n" + "="*70)
        print("🔍 4. گزارش فارسی با RAG")
        print("="*70)
        report_fa_rag = chain_fa_rag.invoke(preliminary_findings)
        report_fa_rag = clean_report_output(report_fa_rag)
        print(report_fa_rag)
        print("------------------------------------------------------------------------------------")
        all_reports += "4. گزارش فارسی با RAG\n" + "="*70 + "\n" + report_fa_rag + "\n" + "-"*84 + "\n\n"
        
        # Save all reports to a file
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"all_reports_{timestamp}.txt"
        
        final_reports_dir = "final-reports"
        os.makedirs(final_reports_dir, exist_ok=True)
        
        file_path = os.path.join(final_reports_dir, filename)
        
        # --- RAG evaluation and report quality ---
        def jaccard_similarity(a, b):
            set_a = set(a.lower().split())
            set_b = set(b.lower().split())
            intersection = set_a & set_b
            union = set_a | set_b
            return len(intersection) / len(union) if union else 0

        def count_medical_terms(text, terms):
            text = text.lower()
            return sum(1 for t in terms if t in text)

        medical_terms = [
            'lung', 'parenchyma', 'mass', 'nodule', 'pleural', 'mediastinal', 'effusion', 'consolidation',
            'lymph', 'adenopathy', 'collapse', 'thorax', 'heart', 'trachea', 'esophagus', 'ریه', 'توده', 'ندول', 'پلور', 'مدیاستن', 'قلب', 'تراشه', 'میدیاستین', 'لنف', 'آدنوپاتی', 'افیوژن', 'کنسولیدیشن', 'کلاپس'
        ]

        # Semantic similarity between NoRAG and RAG reports (English)
        sim_en = jaccard_similarity(report_en_no_rag, report_en_rag)
        # Semantic similarity between NoRAG and RAG reports (Persian)
        sim_fa = jaccard_similarity(report_fa_no_rag, report_fa_rag)
        # Medical term coverage
        med_en_no_rag = count_medical_terms(report_en_no_rag, medical_terms)
        med_en_rag = count_medical_terms(report_en_rag, medical_terms)
        med_fa_no_rag = count_medical_terms(report_fa_no_rag, medical_terms)
        med_fa_rag = count_medical_terms(report_fa_rag, medical_terms)

        eval_text = f"RAG EVALUATION RESULTS\n{'='*70}\n"
        eval_text += f"English Jaccard Similarity (NoRAG vs RAG): {sim_en:.2f}\n"
        eval_text += f"Persian Jaccard Similarity (NoRAG vs RAG): {sim_fa:.2f}\n"
        eval_text += f"Medical Terms in English NoRAG: {med_en_no_rag}\n"
        eval_text += f"Medical Terms in English RAG: {med_en_rag}\n"
        eval_text += f"Medical Terms in Persian NoRAG: {med_fa_no_rag}\n"
        eval_text += f"Medical Terms in Persian RAG: {med_fa_rag}\n"
        eval_text += f"{'='*70}\n"
        print(eval_text)
        all_reports += eval_text

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("CT SCAN REPORTS COMPARISON\n")
                f.write("Generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
                f.write("="*84 + "\n\n")
                f.write(all_reports)
            print(f"\n✅ All reports successfully saved to file: {file_path}")
        except Exception as e:
            print(f"❌ Error saving file: {e}")

if __name__ == "__main__":
    main()
