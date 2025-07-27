import os
import re
import torch
from sentence_transformers import SentenceTransformer, util
from smolagents import tool

# Load a small, efficient embedding model for sentence-level semantic comparison
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Get the absolute path to the current script file
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Define the path to the markdown file
RECIPE_PATH = os.path.join(CURRENT_DIR, "Cook-Book.md")

# --- Helper Function: Extract markdown sections with colons in the headings ---
def extract_sections_with_colon(md_text):
    # Match lines starting with '## ' followed by any text and a ':'
    pattern = r"^## (.+?):"
    # Initialize lists for titles and contents
    titles, contents = [], []
    # Find each header and capture its content until the next header
    for match in re.finditer(pattern, md_text, re.MULTILINE):
        title = match.group(1).strip()
        start = match.end()
        # Search for the next header to determine the end of the current section
        next_match = re.search(pattern, md_text[start:], re.MULTILINE)
        end = start + next_match.start() if next_match else len(md_text)
        # Append title and content to respective list
        contents.append(md_text[start:end].strip())
        titles.append(title)
    return titles, contents


# --- Helper Function: Load and Embed Sections from the Markdown File ---
def load_recipe_sections(filepath):
    # Open and read the file
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    # Extract sections
    titles, contents = extract_sections_with_colon(text)
    if not titles:
        raise ValueError(f"No sections found in {filepath}.")
    # Encode section contents into vector embeddings for semantic search
    embeddings = embedding_model.encode(contents, convert_to_tensor=True)
    return titles, contents, embeddings

# Preprocess and load the storage data at startup
recipe_titles, recipe_texts, recipe_embeddings = load_recipe_sections(RECIPE_PATH)

# Given a user query, retrieve the most semantically similar section from the document
def retrieve_recipe(query):
    # Encode the query
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    # Compute cosine similarity between query and all document sections
    scores = util.cos_sim(query_embedding, recipe_embeddings)[0]
    # Find the index of the highest-scoring section
    best_idx = torch.argmax(scores).item()
    # Return the best-matching section if its similarity score is above threshold
    return recipe_texts[best_idx] if scores[best_idx] > 0.3 else None

# --- Agent Tool ---
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


   
