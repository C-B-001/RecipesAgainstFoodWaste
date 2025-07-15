import os
import re
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from transformers import pipeline
from smolagents import tool

# Load embedding model (better than MiniLM)
embedding_model = SentenceTransformer("BAAI/bge-large-en")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Simple local LLM pipeline for query expansion (can replace with OpenAI)
query_expander = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta")

# File paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RECIPE_PATH = os.path.join(CURRENT_DIR, "Cook-Book.md")


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

# Load and embed a document
def load_document_sections(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    titles, contents = extract_sections(text)
    embeddings = embedding_model.encode(contents, convert_to_tensor=True)
    return titles, contents, embeddings

# Load once and cache
recipe_titles, recipe_texts, recipe_embeddings = load_document_sections(RECIPE_PATH)


# Expand the query using a local or hosted LLM
def expand_query(query: str) -> list[str]:
    prompt = f"""You are helping improve document search. Expand the user query into 3 alternative phrasings or synonyms.
Original: "{query}"
Examples:"""
    output = query_expander(prompt, max_new_tokens=100, do_sample=False)[0]["generated_text"]
    # Parse results - crude filtering
    lines = output.split("\n")
    expansions = [query]
    for line in lines:
        if "-" in line or "•" in line:
            expansions.append(line.split(" ", 1)[-1].strip())
    return list(set(expansions))[:4]  # limit to 4 total variants

# Embed query (with instruction prefix for BGE)
def embed_query(q: str):
    return embedding_model.encode(f"Represent this sentence for searching relevant passages: {q}", convert_to_tensor=True)

# Search and rerank with expanded queries
def retrieve_best_with_rerank(query, titles, texts, embeddings):
    expanded_queries = expand_query(query)
    candidate_sections = []

    # Step 1: Get top 3 for each query variant
    for variant in expanded_queries:
        q_embedding = embed_query(variant)
        scores = util.cos_sim(q_embedding, embeddings)[0]
        top_results = torch.topk(scores, k=min(3, len(scores)))
        for idx in top_results.indices:
            candidate_sections.append((titles[idx], texts[idx], scores[idx].item()))

    # Deduplicate by content
    seen = set()
    unique_candidates = []
    for t, c, s in candidate_sections:
        if c not in seen:
            unique_candidates.append((t, c))
            seen.add(c)

    # Step 2: Rerank using cross-encoder
    pairs = [(query, c) for _, c in unique_candidates]
    if not pairs:
        return None

    scores = cross_encoder.predict(pairs)
    best_idx = int(torch.tensor(scores).argmax())
    return unique_candidates[best_idx][1]

# Recipe search tool

@tool
def search_recipe(query: str) -> str:
    """
    Searches the Cook-Book for a recipe relevant to the query.
    Args:
        query: A short user request or description of a food, dish, or ingredient (e.g. 'overripe banana', 'chocolate cake').
    Returns:
        A text excerpt from the Cook-Book matching the query.
    """
    result = retrieve_best_with_rerank(query, recipe_titles, recipe_texts, recipe_embeddings)
    return result or "Sorry, I couldn’t find a recipe matching your request in the Cook-Book."




