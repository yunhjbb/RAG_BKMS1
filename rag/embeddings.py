from langchain_huggingface.embeddings import HuggingFaceEmbeddings

def load_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    return HuggingFaceEmbeddings(model_name=model_name)
