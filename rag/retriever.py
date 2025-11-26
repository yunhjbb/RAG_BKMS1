def build_retriever(vectorstore, top_k=3):
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": top_k}
    )
