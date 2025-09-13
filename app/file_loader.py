import os
from fastapi import HTTPException
from PyPDF2 import PdfReader
from docx import Document

def read_file(path: str) -> str:
    file = os.path.splitext(path)[1].lower()        # Get file extension (e.g. .txt, .pdf, .docx)
    if file == ".txt":
        with open(path, "r", encoding="utf-8") as file:
            return file.read()
    elif file == ".pdf":
        reader = PdfReader(path)
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    elif file == ".docx":
        doc = Document(path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file}")

# Quick Test

# if __name__ == "__main__":
#     DATA_FOLDER = os.path.join(os.path.dirname(__file__), "..", "data")

#     for fname in os.listdir(DATA_FOLDER):
#         path = os.path.join(DATA_FOLDER, fname)
#         print(read_file(path)[:100])        
#         print(os.path.splitext(fname)[1][1:])         # Get file extension (e.g. .txt, .pdf, .docx)

