import os
import re
import torch
from sentence_transformers import SentenceTransformer, util
from smolagents import tool

# Use a lightweight embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# File paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RECIPE_PATH = os.path.join(CURRENT_DIR, "Cook-Book.md")
STORAGE_PATH = os.path.join(CURRENT_DIR, "Food-Storage.md")

# Extract sections from Markdown using headings
def extract_sections(md_text, heading_pattern=r"^## (.+?):", flags=re.MULTILINE):
    matches = re.findall(heading_pattern, md_text, flags)
    titles, contents = [], []
    for match in re.finditer(heading_pattern, md_text, flags):
        title = match.group(1).strip()
        start = match.end()
        next_match = re.search(heading_pattern, md_text[start:], flags)
        end = start + next_match.start() if next_match else len(md_text)
        content = md_text[start:end].strip()
        titles.append(title)
        contents.append(content)
    return titles, contents

# Load and embed document sections
def load_document_sections(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    titles, contents = extract_sections(text)
    if len(contents) == 0:
        raise ValueError(f"No sections found in {filepath} — please check the file format.")
    embeddings = embedding_model.encode(contents, convert_to_tensor=True)
    return titles, contents, embeddings


# Load once and cache
recipe_titles, recipe_texts, recipe_embeddings = load_document_sections(RECIPE_PATH)
storage_titles, storage_texts, storage_embeddings = load_document_sections(STORAGE_PATH)

# Embed the query
def embed_query(q: str):
    return embedding_model.encode(q, convert_to_tensor=True)

# Retrieve best match based on cosine similarity
def retrieve_best(query, titles, texts, embeddings):
    if embeddings.size(0) == 0:
        return None
    q_embedding = embed_query(query)
    scores = util.cos_sim(q_embedding, embeddings)[0]
    top_results = torch.topk(scores, k=min(3, len(scores)))
    if len(top_results.indices) == 0:
        return None
    best_idx = top_results.indices[0].item()
    return texts[best_idx]

# Recipe search tool
@tool
def search_recipe(query: str) -> str:
    """
    Searches the Cook-Book for a recipe relevant to the query.
    Args:
        query: A short user request or description of a food, dish, or ingredient.
    Returns:
        A text excerpt from the Cook-Book matching the query.
    """
    result = retrieve_best(query, recipe_titles, recipe_texts, recipe_embeddings)
    return result or "Sorry, I couldn’t find a recipe matching your request in the Cook-Book."

# Food storage search tool
@tool
def search_storage(query: str) -> str:
    """
    Searches the Food-Storage guide for storage info.
    Args:
        query: The food item to look up storage info for.
    Returns:
        A relevant section from the Food-Storage guide.
    """
    result = retrieve_best(query, storage_titles, storage_texts, storage_embeddings)
    return result or "Sorry, I couldn’t find any storage advice for that item in the Food-Storage guide."
