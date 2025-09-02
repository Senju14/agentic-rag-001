import re
from bs4 import BeautifulSoup
import os

def clean_text(text: str, file_type: str = "txt") -> str:
    text = text or ""
    if file_type == "html":
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text(separator=" ")
    elif file_type == "md":
        text = re.sub(r'(^|\s)(#{1,6}\s+)', ' ', text)
        text = re.sub(r'(\*\*|\*|`|~~|>)', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()



def load_file(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    file_type = "txt"
    if ext == ".html" or ext == ".htm":
        file_type = "html"
    elif ext == ".md":
        file_type = "md"
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    return clean_text(content, file_type)



def load_all_files(folder_path: str):
    out = []
    for fname in sorted(os.listdir(folder_path)):
        fpath = os.path.join(folder_path, fname)
        if os.path.isfile(fpath) and fname.lower().endswith(('.txt', '.md', '.html', '.htm')):
            ext = os.path.splitext(fname)[1].lower()
            ft = "txt"
            if ext in ('.html', '.htm'):
                ft = "html"
            elif ext == '.md':
                ft = "md"
            text = load_file(fpath)
            out.append((fname, text, ft))
    return out

# if __name__ == "__main__":
#     # quick demo
#     folder = os.path.join(os.path.dirname(__file__), "..", "data")
#     folder = os.path.abspath(folder)
#     for fname, text, ft in load_all_files(folder):
#         print(fname, "->", text[:120].replace("\n", " "), "...\n")
