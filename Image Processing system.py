#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install torch torchvision transformers sentence-transformers faiss-cpu pillow streamlit')


# In[2]:


import os
import torch
import faiss
import numpy as np
import pandas as pd
import clip
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from sentence_transformers import SentenceTransformer


# In[3]:


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


# In[4]:


dataset_path = r"C:\Users\9904136309089\Downloads\Data"

# Define paths for images & CSV file
image_folder = os.path.join(dataset_path, "test_data_v2")
csv_file = os.path.join(dataset_path, "test.csv")

# Load CSV file
df = pd.read_csv(csv_file)
print("CSV Columns:", df.columns)


# In[5]:


import glob

# Check if images exist in the folder
all_images = glob.glob(os.path.join(image_folder, "*"))
print(f"Total images found: {len(all_images)}")
print("Sample images:", all_images[:5])


# In[6]:


def find_correct_file(image_id):
    """Finds correct image file path based on ID"""
    for ext in [".jpg", ".png", ".jpeg"]:
        file_path = os.path.join(image_folder, image_id + ext)
        if os.path.exists(file_path):
            return file_path
    return None  # Return None if file is missing

# Generate valid image paths
image_files = [find_correct_file(str(img_id)) for img_id in df['id']]
image_files = [img for img in image_files if img]  # Remove missing images

print(f"Total valid images: {len(image_files)}")
print("Sample valid image paths:", image_files[:5])


# In[8]:


import os
import glob

# Check if the image folder exists
print("Image Folder Exists:", os.path.exists(image_folder))

# List all image files in the folder
all_images = glob.glob(os.path.join(image_folder, "*"))
print(f"Total images found: {len(all_images)}")
print("Sample image filenames:", all_images[:5])


# In[9]:


print("CSV Columns:", df.columns)
print(df.head())  # Preview first few rows


# In[10]:


print("Sample IDs from CSV:", df['id'].head(10).tolist())


# In[11]:


sample_id = str(df['id'].iloc[0])  # First ID in CSV
for ext in [".jpg", ".png", ".jpeg"]:
    test_path = os.path.join(image_folder, sample_id + ext)
    print(f"Checking: {test_path} -> Exists: {os.path.exists(test_path)}")


# In[12]:


import glob

def find_correct_file(image_id):
    """Find correct file using glob pattern matching"""
    search_path = os.path.join(image_folder, f"{image_id}.*")  # Find any extension
    files = glob.glob(search_path)
    return files[0] if files else None  # Return first match or None

# Generate valid image paths
image_files = [find_correct_file(str(img_id)) for img_id in df['id']]
image_files = [img for img in image_files if img]  # Remove None values

print(f"Total valid images: {len(image_files)}")
print("Sample valid image paths:", image_files[:5])


# In[13]:


image_folder = os.path.join(dataset_path, "test_data_v2")  # ✅ Correct


# In[14]:


import os
import glob

image_folder = r"C:\Users\9904136309089\Downloads\Data\test_data_v2"

# List all images in the folder
image_list = glob.glob(os.path.join(image_folder, "*.jpg"))
print(f"Total Images Found: {len(image_list)}")
print("Sample Image Files:", image_list[:5])


# In[15]:


image_files = [os.path.join(image_folder, img) for img in df['id']]


# In[18]:


import os
import torch
import numpy as np
import pandas as pd
import clip
import faiss
from PIL import Image
from tqdm import tqdm

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Define dataset paths
dataset_path = r"C:\Users\9904136309089\Downloads\Data"
image_folder = os.path.join(dataset_path, "test_data_v2")
csv_file = os.path.join(dataset_path, "test.csv")

# Load CSV file
df = pd.read_csv(csv_file)
print("CSV Columns:", df.columns)
print("Sample IDs:", df['id'].head(5))

# Remove "test_data_v2/" prefix from image IDs
df['id'] = df['id'].str.replace("test_data_v2/", "", regex=False)

# Construct valid image paths
image_files = [os.path.join(image_folder, img) for img in df['id']]

# Verify image paths
valid_images = [img for img in image_files if os.path.exists(img)]
print(f"Total valid images: {len(valid_images)}")
print("Sample valid image paths:", valid_images[:5])

# Compute image embeddings
image_embeddings = []
for image_path in tqdm(valid_images):
    with torch.no_grad():
        image_tensor = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        embedding = model.encode_image(image_tensor).cpu().numpy()
        image_embeddings.append(embedding)

# Check if we have valid embeddings
if len(image_embeddings) > 0:
    image_embeddings = np.vstack(image_embeddings)

    # Save embeddings using FAISS
    dimension = image_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(image_embeddings)

    faiss.write_index(index, "image_index.faiss")
    np.save("image_files.npy", np.array(valid_images))

    print("✅ Image embeddings saved successfully!")
else:
    print("❌ No valid images found! Check file paths.")


# In[23]:


import os
import numpy as np
import faiss
import torch
import clip
import streamlit as st
from PIL import Image

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load FAISS index and image list
index = faiss.read_index("image_index.faiss")
image_files = np.load("image_files.npy", allow_pickle=True)

# Define the search function
def search_images(query, top_k=5):
    """Search for images matching a text query."""
    text_tokens = clip.tokenize([query]).to(device)
    
    with torch.no_grad():
        text_embedding = model.encode_text(text_tokens).cpu().numpy()

    # Search in FAISS index
    distances, indices = index.search(text_embedding, top_k)

    # Retrieve image paths
    results = [image_files[i] for i in indices[0]]
    return results


# In[ ]:


st.title("Multi-Modal Image Retrieval System")

query = st.text_input("Enter a text description:")

if st.button("Search"):
    if query:
        results = search_images(query)  # Now correctly defined
        for img_path in results:
            st.image(img_path, caption=os.path.basename(img_path), use_column_width=True)
    else:
        st.warning("Please enter a query.")


# In[24]:


pip install clip-by-openai


# In[26]:


import os
import torch
import numpy as np
import faiss
import clip
from PIL import Image
from tqdm import tqdm
import pandas as pd

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Define dataset paths
dataset_path = r"C:\Users\9904136309089\Downloads\Data"
image_folder = os.path.join(dataset_path, "test_data_v2")
csv_file = os.path.join(dataset_path, "test.csv")

# Load CSV file
df = pd.read_csv(csv_file)

# Remove "test_data_v2/" prefix from filenames if needed
df['id'] = df['id'].str.replace("test_data_v2/", "", regex=False)

# Construct valid image paths
image_files = [os.path.join(image_folder, img) for img in df['id']]
valid_images = [img for img in image_files if os.path.exists(img)]

# Select a random sample of 500 images (if more exist)
if len(valid_images) > 500:
    valid_images = np.random.choice(valid_images, 500, replace=False)

print(f"Total valid images (limited to 500): {len(valid_images)}")
print("Sample valid image paths:", valid_images[:5])

# Compute image embeddings
image_embeddings = []
for image_path in tqdm(valid_images):
    with torch.no_grad():
        image_tensor = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        embedding = model.encode_image(image_tensor).cpu().numpy()
        image_embeddings.append(embedding)

# Check if embeddings exist before saving
if len(image_embeddings) > 0:
    image_embeddings = np.vstack(image_embeddings)

    # Save embeddings using FAISS
    dimension = image_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(image_embeddings)

    # Save index to file
    faiss.write_index(index, os.path.join(dataset_path, "image_index.faiss"))
    np.save(os.path.join(dataset_path, "image_files.npy"), np.array(valid_images))

    print("✅ Image embeddings saved successfully!")
else:
    print("❌ No valid images found! Check file paths.")


# In[2]:


import faiss
import os
import numpy as np

dataset_path = r"C:\Users\9904136309089\Downloads\Data"
index_path = os.path.join(dataset_path, "image_index.faiss")
image_files_path = os.path.join(dataset_path, "image_files.npy")

# Load FAISS index and image file list
index = faiss.read_index(index_path)
image_files = np.load(image_files_path, allow_pickle=True)

print(f"Total images in FAISS index: {index.ntotal}")
print(f"Total images in image_files.npy: {len(image_files)}")

# Check first 5 images
print("Sample image paths:", image_files[:5])


# In[ ]:




