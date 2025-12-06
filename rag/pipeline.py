from .loaders import load_and_split_pdfs
from .embeddings import load_embedding_model
from .vectorstore import build_or_load_chroma, build_or_load_faiss
from .retriever import MultiRetriever
from .agent import build_agent

def build_rag_pipeline(model):

    print("Loading PDFs and splitting...")
    manu_chunks, ref_chunks, map = load_and_split_pdfs(
        cache_manu="./cache/manu.pkl",
        cache_refs="./cache/refs.pkl"
    )

    print("Loading embedding model...")
    embed = load_embedding_model()

    print("Building / Loading Chroma (manuscript only)...")
    chroma_db = build_or_load_chroma(
        manu_chunks, embed, "./chroma_store"
    )

    print("Building / Loading FAISS (reference only)...")
    faiss_db = build_or_load_faiss(
        ref_chunks, embed, "./faiss_store"
    )

    print("Creating retrievers...")
    chroma_retriever = chroma_db.as_retriever(search_kwargs={"k": 5})
    faiss_retriever  = faiss_db.as_retriever(search_kwargs={"k": 5})

    retriever = MultiRetriever(chroma_retriever, faiss_retriever, map)

    print("Creating agent...")
    agent = build_agent(model, retriever)

    return agent
