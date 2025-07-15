import os
import re
import torch
from sentence_transformers import SentenceTransformer, util
from smolagents import tool

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_PATH = os.path.join(CURRENT_DIR, "Food-Storage.md")

def extract_sections_without_colon(md_text):
    pattern = r"^## (.+)$"
    titles, contents = [], []
    for match in re.finditer(pattern, md_text, re.MULTILINE):
        title = match.group(1).strip()
        start = match.end()
        next_match = re.search(pattern, md_text[start:], re.MULTILINE)
        end = start + next_match.start() if next_match else len(md_text)
        contents.append(md_text[start:end].strip())
        titles.append(title)
    return titles, contents

def load_storage_sections(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    titles, contents = extract_sections_without_colon(text)
    if not titles:
        raise ValueError(f"No sections found in {filepath}.")
    embeddings = embedding_model.encode(contents, convert_to_tensor=True)
    return titles, contents, embeddings

storage_titles, storage_texts, storage_embeddings = load_storage_sections(STORAGE_PATH)

def retrieve_storage(query):
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, storage_embeddings)[0]
    best_idx = torch.argmax(scores).item()
    return storage_texts[best_idx] if scores[best_idx] > 0.3 else None

@tool
def search_storage(query: str) -> str:
    """Searches the Food-Storage guide for storage advice relevant to the query.
    Args:
        query: The user's request (e.g., a vegetable or fruit name).
    """
    result = retrieve_storage(query)
    if result:
        return result
    else:
        return "Sorry, I couldn’t find information in the food storage document."

