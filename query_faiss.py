# query_faiss.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# 初始化：只在第一次調用時載入
model = None
index = None
documents = None

def load_resources():
    global model, index, documents
    if model is None:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    if index is None:
        index = faiss.read_index("nutrition_index.faiss")
    if documents is None:
        with open("nutrition_docs.pkl", "rb") as f:
            documents = pickle.load(f)

def search_similar_documents(query: str, top_k: int = 5) -> str:
    load_resources()
    embedding = model.encode([query])
    D, I = index.search(np.array(embedding), k=top_k)
    relevant_docs = [documents[i] for i in I[0]]
    return "\n".join(relevant_docs)
