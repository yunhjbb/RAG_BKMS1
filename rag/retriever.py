class MultiRetriever:
    def __init__(self, manuscript_retriever, reference_retriever, reference_index_map=None):
        """
        manuscript_retriever: manuscript 전용 retriever
        reference_retriever: reference 전용 retriever
        reference_index_map: { "1": [ref_chunk1, ref_chunk2], ... }  (direct lookup mode)
        """
        self.manuscript = manuscript_retriever
        self.reference = reference_retriever
        self.ref_index = reference_index_map or {}

    # -------------------------------
    # LCEL이 호출하는 기본 검색 기능
    # -------------------------------
    def invoke(self, query, **kwargs):
        if isinstance(query, dict):
            query = query.get("query") or query.get("input")
        return self.manuscript.invoke(query)

    # -------------------------------
    # Reference-aware 검색 기능
    # -------------------------------
    def search_reference_by_index(self, cited_keys_str):
        """
        cited_keys_str: "1,3,7" 또는 "" 형태의 문자열
        """
        if not cited_keys_str:
            return []

        # 문자열 → 리스트 변환
        cited_keys = [key.strip() for key in cited_keys_str.split(",") if key.strip().isdigit()]

        results = []
        for cid in cited_keys:
            if cid in self.ref_index:
                results.extend(self.ref_index[cid])
        return results

    def search_reference_semantic(self, title, k=2):
        """fallback으로 semantic search도 가능"""
        return self.reference.invoke(title)[:k]
