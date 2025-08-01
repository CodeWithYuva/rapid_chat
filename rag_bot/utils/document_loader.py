import docx
import fitz  # PyMuPDF
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import torch


model_id = "microsoft/phi-1_5"
local_model_path = "local_models/phi-1_5"

# Check if the model/tokenizer is already saved locally
def is_model_cached(path):
    return os.path.isdir(path) 

# Load and optionally save
if is_model_cached(local_model_path):
    print(" Using local model files...")
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
    llm = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        torch_dtype=torch.float32,
        local_files_only=True,
        device_map={"": "cpu"},
        do_sample=False
    )
else:
    print(" Downloading model from Hugging Face and saving locally...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    llm = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map={"": "cpu"},
        do_sample=False
    )

    # Try saving locally
    try:
        llm.save_pretrained(local_model_path)
        tokenizer.save_pretrained(local_model_path)
        print(" Model saved to local_models/phi-1_5")
    except Exception as e:
        print(f" Could not save model: {e}")



MAX_TOKENS = 100
MIN_TOKENS = 60
OVERLAP = 20

def tokenize_length(text):
    return len(tokenizer.tokenize(text))

def split_paragraph(paragraph, max_tokens=MAX_TOKENS, overlap=OVERLAP):
    tokens = tokenizer.tokenize(paragraph)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens).strip()
        chunks.append(chunk_text)
        start += max_tokens - overlap
    return chunks

def smart_chunk_paragraphs(paragraphs):
    result_chunks = []
    buffer = ""
    
    i = 0
    while i < len(paragraphs):
        para = paragraphs[i].strip()
        if not para:
            i += 1
            continue

        token_len = tokenize_length(para)

        if token_len < MIN_TOKENS:
            # Combine small paragraph with next one(s)
            buffer += (" " + para).strip()
            i += 1
            continue

        if buffer:
            # Combine buffer + current if combined length is acceptable
            combined = buffer + " " + para
            combined_token_len = tokenize_length(combined)
            if combined_token_len <= MAX_TOKENS:
                result_chunks.append(combined.strip())
            else:
                # Add buffer alone, then process current
                result_chunks.append(buffer.strip())
                if token_len > MAX_TOKENS:
                    result_chunks.extend(split_paragraph(para))
                else:
                    result_chunks.append(para)
            buffer = ""
        else:
            if token_len > MAX_TOKENS:
                result_chunks.extend(split_paragraph(para))
            else:
                result_chunks.append(para)
        i += 1

    if buffer:
        if tokenize_length(buffer) > MAX_TOKENS:
            result_chunks.extend(split_paragraph(buffer))
        else:
            result_chunks.append(buffer.strip())

    return result_chunks

def process_docx(file, name="docx_file"):
    doc = docx.Document(file)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    raw_chunks = smart_chunk_paragraphs(paragraphs)

    final = []
    for i, text in enumerate(raw_chunks):
        final.append({
            "text": text,
            "metadata": {
                "source": name,
                "page": 1,
                "chunk_id": f"c{i+1}"
            }
        })

    print(f" Smart-chunked into {len(final)} chunks (~{MAX_TOKENS} tokens each).")
    
    return final

def process_pdf(file, name="pdf_file"):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    all_chunks = []

    for page_num in range(len(doc)):
        text = doc[page_num].get_text()
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        page_chunks = smart_chunk_paragraphs(paragraphs)
        for i, chunk in enumerate(page_chunks):
            all_chunks.append({
                "text": chunk,
                "metadata": {
                    "source": name,
                    "page": page_num + 1,
                    "chunk_id": f"p{page_num+1}_c{i+1}"
                }
            })

    print(f" Smart-chunked into {len(all_chunks)} chunks from PDF (~{MAX_TOKENS} tokens each).")
    
    return all_chunks

def process_file(file):
    name = getattr(file, "name", "uploaded_file")
    if name.endswith(".docx"):
        return process_docx(file, name)
    elif name.endswith(".pdf"):
        return process_pdf(file, name)
    else:
        print("❌ Unsupported file format")
        return []
