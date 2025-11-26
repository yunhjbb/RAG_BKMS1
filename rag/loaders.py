import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split_pdfs(pdf_glob="./data/pdfs/*.pdf"):
    all_docs = []

    for pdf_path in glob.glob(pdf_glob):
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        for d in docs:
            d.metadata["source"] = pdf_path

        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(all_docs)
    return chunks
