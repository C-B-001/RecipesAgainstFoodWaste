import re
import os
import torch
from sentence_transformers import SentenceTransformer, util
from smolagents import tool

# Initialize embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# File paths (assuming this file is inside "tools/" directory)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RECIPE_PATH = os.path.join(CURRENT_DIR, "Cook-Book.md")
STORAGE_PATH = os.path.join(CURRENT_DIR, "Food-Storage.md")

# Helper: extract sections using markdown heading syntax
def extract_sections(md_text, heading_pattern=r"^## (.+?):", flags=re.MULTILINE):
    matches = re.findall(heading_pattern, md_text, flags)
    titles = []
    contents = []
    for match in re.finditer(heading_pattern, md_text, flags):
        title = match.group(1).strip()
        start = match.end()
        next_match = re.search(heading_pattern, md_text[start:], flags)
        end = start + next_match.start() if next_match else len(md_text)
        content = md_text[start:end].strip()
        titles.append(title)
        contents.append(content)
    return titles, contents

# Load and encode document sections
def load_document_sections(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    titles, contents = extract_sections(text)
    embeddings = embedding_model.encode(contents, convert_to_tensor=True)
    return titles, contents, embeddings

# Load documents once (cached)
recipe_titles, recipe_texts, recipe_embeddings = load_document_sections(RECIPE_PATH)
storage_titles, storage_texts, storage_embeddings = load_document_sections(STORAGE_PATH)

# Retrieval with hybrid matching
def retrieve_best_match(query, titles, texts, embeddings, top_k=1):
    query_lower = query.lower()
    # 1. Check for exact or partial title match
    for i, title in enumerate(titles):
        if query_lower in title.lower():
            return texts[i]

    # 2. Use embedding similarity
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    if embeddings.shape[0] == 0:
        return None

    scores = util.cos_sim(query_embedding, embeddings)[0]
    top_idx = torch.argmax(scores).item()
    if scores[top_idx] < 0.3:  # similarity threshold
        return None

    return texts[top_idx]

# Tool: search for a recipe
@tool
def search_recipe(query: str) -> str:
    """Searches the Cook-Book for a recipe relevant to the query.
    Args:
        query: The user's request (e.g., an ingredient or dish name).
    """
    result = retrieve_best_match(query, recipe_titles, recipe_texts, recipe_embeddings)
    if result:
        return result
    else:
        return "Sorry, I couldn’t find a recipe matching your request in the Cook-Book."

# Tool: search for food storage info
@tool
def search_storage(query: str) -> str:
    """Searches the Food-Storage guide for how to store a given item.
    Args:
        query: The item to look up (e.g., 'basil', 'bananas').
    """
    result = retrieve_best_match(query, storage_titles, storage_texts, storage_embeddings)
    if result:
        return result
    else:
        return "Sorry, I couldn’t find any storage advice for that item in the Food-Storage guide."
