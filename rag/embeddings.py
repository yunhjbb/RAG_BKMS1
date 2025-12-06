from langchain_huggingface.embeddings import HuggingFaceEmbeddings

def load_embedding_model(
    model_name="hkunlp/instructor-base",
    device="cuda"   # GPU 사용 (없으면 "cpu")
):
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}  # 검색 성능 향상 옵션
    )