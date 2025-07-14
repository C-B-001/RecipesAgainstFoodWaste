import re
import torch
from sentence_transformers import SentenceTransformer, util
from smolagents import tool

# Load embedding model once
embedding_model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

# ========== Function to extract sections based on markdown headings ==========
def extract_sections_from_markdown(md_text, heading_marker="##"):
    pattern = rf"{heading_marker} (.*?):?\n(.*?)(?=\n{heading_marker} |\Z)"
    return re.findall(pattern, md_text, flags=re.DOTALL)

# ========== Load and index Cook-Book ==========
with open("Cook-Book.md", encoding="utf-8") as f:
    cookbook_text = f.read()

cookbook_sections = extract_sections_from_markdown(cookbook_text)
cookbook_titles = [title.strip() for title, _ in cookbook_sections]
cookbook_contents = [content.strip() for _, content in cookbook_sections]
cookbook_embeddings = embedding_model.encode(cookbook_contents, convert_to_tensor=True)

# ========== Load and index Food-Storage ==========
with open("Food-Storage.md", encoding="utf-8") as f:
    storage_text = f.read()

storage_sections = extract_sections_from_markdown(storage_text)
storage_titles = [title.strip() for title, _ in storage_sections]
storage_contents = [content.strip() for _, content in storage_sections]
storage_embeddings = embedding_model.encode(storage_contents, convert_to_tensor=True)

# ========== Cook-Book Search Tool ==========
@tool
def search_recipe(query: str) -> str:
    """
    A tool that searches the Cook-Book for a relevant recipe.
    Args:
        query: A cooking-related question.
    """
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, cookbook_embeddings)[0]
    top_idx = torch.argmax(scores).item()

    title = cookbook_titles[top_idx]
    content = cookbook_contents[top_idx]

    return f"## {title}:\n\n{content}"

# ========== Food-Storage Search Tool ==========
@tool
def search_storage(query: str) -> str:
    """
    A tool that searches the Food-Storage guide for preservation advice.
    Args:
        query: A food item or storage-related question.
    """
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, storage_embeddings)[0]
    top_idx = torch.argmax(scores).item()

    title = storage_titles[top_idx]
    content = storage_contents[top_idx]

    return f"## {title}:\n\n{content}"
