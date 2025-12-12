import os
import glob
import pickle
import json
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter


# ----------------------------------------------------------
# Helper: citation extraction
# ----------------------------------------------------------
def extract_citation_numbers(text):
    pattern = r"\[\s*(\d+(?:\s*[-–]\s*\d+|\s*,\s*\d+)*)\s*\]"
    matches = re.findall(pattern, text)

    result = set()
    for m in matches:
        parts = re.split(r"[, ]+", m)
        for p in parts:
            p = p.strip()

            if "-" in p or "–" in p:
                p = p.replace("–", "-")
                a, b = map(int, p.split("-"))
                result.update(str(i) for i in range(a, b+1))
            elif p.isdigit():
                result.add(p)

    return sorted(list(result), key=lambda x: int(x))


# ----------------------------------------------------------
# Pickle helpers
# ----------------------------------------------------------
def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ----------------------------------------------------------
# Main loader
# ----------------------------------------------------------
def load_and_split_pdfs(
    pdf_glob="./data/pdfs/*.pdf",
    manuscript_pattern="manuscript",
    reference_json_path="./data/references_map.json",
    cache_manu=None,
    cache_refs=None
):
    """
    Returns:
        manuscript_chunks, reference_chunks
    """

    # ----------------------------------------------------------
    # Load cache
    # ----------------------------------------------------------
    if cache_manu and cache_refs and os.path.exists(cache_manu) and os.path.exists(cache_refs):
        print(f"[Cache] Loading manuscript chunks from {cache_manu}")
        manu_chunks = load_pickle(cache_manu)

        print(f"[Cache] Loading reference chunks from {cache_refs}")
        ref_chunks = load_pickle(cache_refs)

    
        reference_index_map = {}
        for ch in ref_chunks:
            idx = str(ch.metadata.get("ref_index"))
            if idx:
                reference_index_map.setdefault(idx, []).append(ch)

        return manu_chunks, ref_chunks, reference_index_map

    # ----------------------------------------------------------
    # Load reference JSON
    # ----------------------------------------------------------
    with open(reference_json_path, "r", encoding="utf-8") as f:
        ref_list = json.load(f)
    ref_map = {str(entry["index"]): entry for entry in ref_list}

    print(f"[References] Loaded {len(ref_map)} entries")

    # ----------------------------------------------------------
    # Load PDFs
    # ----------------------------------------------------------
    manuscript_docs = []
    reference_docs  = []

    for pdf_path in glob.glob(pdf_glob):
        filename = os.path.basename(pdf_path)
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        is_manuscript = manuscript_pattern.lower() in filename.lower()

        # ---- assign metadata to each page ----
        for p in pages:
            p.metadata["source"] = filename
            p.metadata["type"]   = "manuscript" if is_manuscript else "reference"

        # ---- match reference files ----
        if not is_manuscript:
            matched = None
            for ref in ref_list:
                if not ref["local_pdf_path"]:
                    continue

                json_filename = os.path.basename(ref["local_pdf_path"].replace("\\", "/"))

                if json_filename == filename:
                    matched = ref
                    break

            if matched:
                for p in pages:
                    p.metadata["ref_index"] = matched["index"]
                    p.metadata["ref_title"] = matched["title"]
                    p.metadata["ref_raw"]   = matched["raw"]
                    p.metadata["ref_json"]  = matched

        # ---- store docs ----
        if is_manuscript:
            manuscript_docs.extend(pages)
        else:
            reference_docs.extend(pages)

    # ----------------------------------------------------------
    # Chunking
    # ----------------------------------------------------------
    manu_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )

    ref_splitter = TokenTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    manu_chunks = manu_splitter.split_documents(manuscript_docs)
    ref_chunks  = ref_splitter.split_documents(reference_docs)

    print(f"[Chunks] Manuscript chunks: {len(manu_chunks)}")
    print(f"[Chunks] Reference chunks:  {len(ref_chunks)}")

    # ----------------------------------------------------------
    # Attach citation metadata to manuscript chunks
    # ----------------------------------------------------------
    for ch in manu_chunks:
        ids = extract_citation_numbers(ch.page_content)
        ch.metadata["cited_keys"] = ",".join(ids) if ids else ""
        ch.metadata["cited_titles"] = ";".join(
            ref_map[i]["title"] for i in ids if i in ref_map
        )

    # reference chunks do not cite anything
    for ch in ref_chunks:
        ch.metadata["cited_keys"] = []
        ch.metadata["cited_refs"] = []
    reference_index_map = {}

    for ch in ref_chunks:
        idx = str(ch.metadata.get("ref_index"))
        if idx:
            reference_index_map.setdefault(idx, []).append(ch)
    # ----------------------------------------------------------
    # Save cache
    # ----------------------------------------------------------
    if cache_manu:
        os.makedirs(os.path.dirname(cache_manu), exist_ok=True)
        save_pickle(manu_chunks, cache_manu)

    if cache_refs:
        os.makedirs(os.path.dirname(cache_refs), exist_ok=True)
        save_pickle(ref_chunks, cache_refs)

    return manu_chunks, ref_chunks, reference_index_map
