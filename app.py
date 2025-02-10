import os
import numpy as np
import faiss
import torch
import clip
import streamlit as st
from PIL import Image

# Fix OpenMP error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load FAISS index and image list (ensure correct path)
dataset_path = r"C:\Users\9904136309089\Downloads\Data"
index_path = os.path.join(dataset_path, "image_index.faiss")
image_files_path = os.path.join(dataset_path, "image_files.npy")

# Check if FAISS index and image list exist
if not os.path.exists(index_path) or not os.path.exists(image_files_path):
    st.error("❌ Error: FAISS index or image list not found! Run the embedding script first.")
else:
    index = faiss.read_index(index_path)
    image_files = np.load(image_files_path, allow_pickle=True)


# Define the search function
def search_images(query, top_k=5):
    """Search for images matching a text query and return correct results."""
    query = query.strip().lower()  # Normalize text

    # Tokenize and encode text
    text_tokens = clip.tokenize([query]).to(device)
    with torch.no_grad():
        text_embedding = model.encode_text(text_tokens).cpu().numpy()

    # Search in FAISS index
    distances, indices = index.search(text_embedding, top_k)

    # Retrieve correct image paths
    results = []
    for i in indices[0]:
        if 0 <= i < len(image_files):  # Ensure valid index
            results.append(image_files[i])

    return results

# Streamlit UI
st.title("🔍 Multi-Modal Image Retrieval System")

query = st.text_input("Enter a text description:")

if st.button("Search"):
    if query:
        results = search_images(query)
        if results:
            for img_path in results:
                if os.path.exists(img_path):
                    st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)
                else:
                    st.warning(f"⚠️ Image not found: {img_path}")
        else:
            st.error("❌ No matching images found.")
    else:
        st.warning("⚠️ Please enter a query.")
