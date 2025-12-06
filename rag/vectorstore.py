import os
from langchain_community.vectorstores import FAISS
import chromadb
from langchain_community.vectorstores import Chroma

def build_chroma(chunks, embedding_model, collection_name="rag_collection", path="./chroma_store"):
    client = chromadb.PersistentClient(path=path)
    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=collection_name,
        client=client
    )

def load_chroma(collection_name="rag_collection", path="./chroma_store", embedding_model=None):
    client = chromadb.PersistentClient(path=path)
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        client=client
    )

def build_or_load_chroma(chunks, embedding_model, path="./chroma_store", collection_name="rag_collection"):
    # 조건: path 폴더가 있으면 이미 DB 있음 (컬렉션 자동 로드됨)
    if os.path.exists(path) and os.listdir(path):
        print(f"[Chroma] Loading existing Chroma DB from {path}")
        return load_chroma(collection_name, path, embedding_model)

    print("[Chroma] Building a new Chroma DB...")
    db = build_chroma(chunks, embedding_model, collection_name, path)
    print("[Chroma] Saved Chroma DB.")
    return db

def build_faiss(chunks, embedding_model):
    return FAISS.from_documents(chunks, embedding_model)

def save_faiss(vectorstore, path="./faiss_store"):
    vectorstore.save_local(path)

def load_faiss(path, embedding_model):
    return FAISS.load_local(path, embeddings=embedding_model, allow_dangerous_deserialization=True)

def build_or_load_faiss(chunks, embedding_model, path="./faiss_store"):
    if os.path.exists(path):
        print(f"[FAISS] Loading existing FAISS index from {path}")
        return load_faiss(path, embedding_model)

    print("[FAISS] Building a new FAISS index...")
    db = build_faiss(chunks, embedding_model)
    save_faiss(db, path)
    print("[FAISS] Saved FAISS index.")
    return db