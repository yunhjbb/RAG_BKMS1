from .loaders import load_and_split_pdfs
from .embeddings import load_embedding_model
from .vectorstore import build_faiss
from .retriever import build_retriever
from .agent import build_agent

def build_rag_pipeline(model):

    print("Loading PDFs and splitting...")
    chunks = load_and_split_pdfs()

    print("Loading embedding model...")
    embed = load_embedding_model()

    print("Building FAISS DB...")
    vectorstore = build_faiss(chunks, embed)

    print("Creating retriever...")
    retriever = build_retriever(vectorstore)

    print("Creating agent...")
    agent = build_agent(model, retriever)

    return agent
