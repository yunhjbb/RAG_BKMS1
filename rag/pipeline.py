from .loaders import load_and_split_pdfs
from .embeddings import load_embedding_model
from .vectorstore import build_or_load_chroma, build_or_load_faiss
from .retriever import MultiRetriever
from .agent import build_agent
import os

def build_rag_pipeline(model, chunk_size_manu, chunk_overlap_manu, \
                               chunk_size_ref, chunk_overlap_ref):

    base_dir = (
        f"./runs/"
        f"manu_cs{chunk_size_manu}_co{chunk_overlap_manu}"
        f"__ref_cs{chunk_size_ref}_co{chunk_overlap_ref}"
    )

    cache_dir  = os.path.join(base_dir, "cache")
    chroma_dir = os.path.join(base_dir, "chroma_store")
    faiss_dir  = os.path.join(base_dir, "faiss_store")

    os.makedirs(cache_dir, exist_ok=True)
    print("Loading PDFs and splitting...")
    manu_chunks, ref_chunks, map = load_and_split_pdfs(
        cache_manu=os.path.join(cache_dir, "manu.pkl"),
        cache_refs=os.path.join(cache_dir, "refs.pkl"),
    )

    print("Loading embedding model...")
    embed = load_embedding_model()

    print("Building / Loading Chroma (manuscript only)...")
    chroma_db = build_or_load_chroma(
        manu_chunks, embed, path=chroma_dir
    )

    print("Building / Loading FAISS (reference only)...")
    faiss_db = build_or_load_faiss(
        ref_chunks, embed, path=faiss_dir
    )

    print("Creating retrievers...")
    chroma_retriever = chroma_db.as_retriever(search_kwargs={"k": 5})
    faiss_retriever  = faiss_db.as_retriever(search_kwargs={"k": 5})

    retriever = MultiRetriever(chroma_retriever, faiss_retriever, map)

    print("Creating agent...")
    agent = build_agent(model, retriever)

    return agent
