import os
import re
import torch
from sentence_transformers import SentenceTransformer, util
from smolagents import tool

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RECIPE_PATH = os.path.join(CURRENT_DIR, "Cook-Book.md")

def extract_sections_with_colon(md_text):
    pattern = r"^## (.+?):"
    titles, contents = [], []
    for match in re.finditer(pattern, md_text, re.MULTILINE):
        title = match.group(1).strip()
        start = match.end()
        next_match = re.search(pattern, md_text[start:], re.MULTILINE)
        end = start + next_match.start() if next_match else len(md_text)
        contents.append(md_text[start:end].strip())
        titles.append(title)
    return titles, contents

def load_recipe_sections(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    titles, contents = extract_sections_with_colon(text)
    if not titles:
        raise ValueError(f"No sections found in {filepath}.")
    embeddings = embedding_model.encode(contents, convert_to_tensor=True)
    return titles, contents, embeddings

recipe_titles, recipe_texts, recipe_embeddings = load_recipe_sections(RECIPE_PATH)

def retrieve_recipe(query):
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, recipe_embeddings)[0]
    best_idx = torch.argmax(scores).item()
    return recipe_texts[best_idx] if scores[best_idx] > 0.3 else None

@tool
def search_recipe(query: str) -> str:
    """Searches the Cook-Book for a recipe relevant to the query.
    Args:
        query: The user's request (e.g., an ingredient or dish name).
    """
    result = retrieve_recipe(query)
    if result:
        return result
    else:
        return "Sorry, I couldn’t find a recipe matching your request in the Cook-Book."


   
