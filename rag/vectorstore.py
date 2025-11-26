from langchain_community.vectorstores import FAISS

def build_faiss(chunks, embedding_model):
    return FAISS.from_documents(chunks, embedding_model)

def save_faiss(vectorstore, path="./faiss_store"):
    vectorstore.save_local(path)

def load_faiss(path, embedding_model):
    return FAISS.load_local(path, embedding_model)
