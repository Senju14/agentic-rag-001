import os
from fastapi import HTTPException
from docx import Document
import pdfplumber 

def read_file(path: str) -> str:
    file = os.path.splitext(path)[1].lower()        # Get file extension (e.g. .txt, .pdf, .docx)
    if file == ".txt":
        with open(path, "r", encoding="utf-8") as file:
            return file.read()
    elif file == ".pdf":
        texts = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                texts.append(page.extract_text() or "")
        return "\n".join(texts)
    elif file == ".docx":
        doc = Document(path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file}")
 
# Quick Test
if __name__ == "__main__":
    DATA_FOLDER = os.path.join(os.path.dirname(__file__), "..", "data")
    for fname in os.listdir(DATA_FOLDER):
        path = os.path.join(DATA_FOLDER, fname)
        print(read_file(path)[:1000])        
        print(os.path.splitext(fname)[1])         # Get file extension (e.g. .txt, .pdf, .docx)
