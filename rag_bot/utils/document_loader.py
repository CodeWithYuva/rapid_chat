import fitz
import docx

def process_pdf(file, name):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    chunks = []
    for page_num in range(len(doc)):
        text = doc[page_num].get_text()
        for i, chunk in enumerate(text.split("\n\n")):
            if chunk.strip():
                chunks.append({
                    "text": chunk.strip(),
                    "metadata": {
                        "source": name,
                        "page": page_num + 1,
                        "chunk_id": f"p{page_num+1}_c{i+1}"
                    }
                })
    return chunks

def process_docx(file, name):
    doc = docx.Document(file)
    return [
        {
            "text": para.text.strip(),
            "metadata": {
                "source": name,
                "page": 1,
                "chunk_id": f"c{i+1}"
            }
        }
        for i, para in enumerate(doc.paragraphs) if para.text.strip()
    ]

def process_file(file):
    name = file.name
    if name.endswith(".pdf"):
        return process_pdf(file, name)
    elif name.endswith(".docx"):
        return process_docx(file, name)
    return []
