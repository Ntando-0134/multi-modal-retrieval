# 📷 Multi-Modal Image Retrieval System

A multi-modal retrieval system that allows users to search images using text descriptions, leveraging **CLIP, FAISS, and Streamlit**.

---

## 🚀 Features
- **Natural language search**: Retrieve images using descriptive text.
- **CLIP model**: Generates embeddings for both text and images.
- **FAISS index**: Enables fast nearest-neighbor search.
- **Streamlit UI**: Simple web interface for user interaction.

---

## 📥 Installation
### 1️⃣ Clone the Repository
```sh
git clone https://github.com/Ntando-0134/multi-modal-retrieval.git
cd multi-modal-retrieval

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Generate Image Embeddings
python embeddings.py

4️⃣ Run the Streamlit App
streamlit run app.py

🛠️ Running Tests
pytest tests/
