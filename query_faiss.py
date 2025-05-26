# query_faiss.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os
# 初始化：只在第一次調用時載入
model = None
index = None
documents = None

def load_resources():
    global model, index, documents
    print("📂 當前目錄：", os.getcwd())
    print("📁 檔案列表：", os.listdir("."))

    if model is None:
        model = SentenceTransformer('all-MiniLM-L6-v2')

    if index is None:
        try:
            print("Loading faiss...")
            index = faiss.read_index("nutrition_index.faiss")
            print("faiss success")
        except Exception as e:
            print("❌ FAISS 載入失敗：", e)

    if documents is None:
        try:
            print("Loading pkl")
            with open("nutrition_docs.pkl", "rb") as f:
                documents = pickle.load(f)
            print("pkl success")
        except Exception as e:
            print("❌ PKL 載入失敗：", e)

def search_similar_documents(query: str, top_k: int = 10) -> str:
    print("🔍 呼叫 search_similar_documents()")
    load_resources()
    embedding = model.encode([query])
    D, I = index.search(np.array(embedding), k=top_k)
    relevant_docs = [documents[i] for i in I[0]]
    print("查詢 query：", query)
    print("找到文件：", relevant_docs)
    return "\n".join(relevant_docs)
