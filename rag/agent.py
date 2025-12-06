from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain_core.prompts import ChatPromptTemplate


def build_agent(model, retriever):

    # -------------------------------
    # (1) Query rewriting
    # -------------------------------
    def rewrite_query(model, original_query: str) -> str:
        rewrite_system_msg = (
            "You are an expert query rewriter. Rewrite the user's question into a "
            "concise, precise, search-optimized query for similarity search. "
            "Return ONLY the rewritten query."
        )

        prompt = ChatPromptTemplate(
            [
                ("system", rewrite_system_msg),
                ("human", "{user_query}")
            ]
        )

        prompt_value = prompt.invoke({"user_query": original_query})
        response = model.invoke(prompt_value.messages)

        return response.content.strip()

    # -------------------------------
    # (2) Reference-aware retrieval
    # -------------------------------
    def expand_with_references(docs, retriever, top_k_ref=2, semantic_fallback=False):
        """
        docs: manuscript 검색 결과
        retriever: MultiRetriever
        semantic_fallback: title 기반 semantic search를 fallback으로 사용할지 여부
        """

        extra_ref_docs = []

        for d in docs:
            cited_keys_str = d.metadata.get("cited_keys", "")

            # 1) 번호 기반 reference lookup
            direct_refs = retriever.search_reference_by_index(cited_keys_str)
            if direct_refs:
                ref_ids = sorted({doc.metadata.get("ref_index") for doc in direct_refs})
                print(f"[Ref Lookup] cited_keys={cited_keys_str} -> found reference indices: {ref_ids}")
            else:
                print(f"[Ref Lookup] cited_keys={cited_keys_str} -> no references found")
            extra_ref_docs.extend(direct_refs)

            # 2) fallback: cited_refs에서 title 얻어 semantic 검색 (기본 False)
            if semantic_fallback:
                cited_info = d.metadata.get("cited_refs", [])
                for ref_entry in cited_info:
                    ref_title = ref_entry.get("title") or ref_entry.get("title_from_ref")
                    if not ref_title:
                        continue
                    extra_ref_docs.extend(
                        retriever.search_reference_semantic(ref_title, k=top_k_ref)
                    )

        return docs + extra_ref_docs


    # -------------------------------
    # (3) Middleware
    # -------------------------------
    @dynamic_prompt
    def middleware_prompt(request: ModelRequest) -> str:
        user_query = request.state["messages"][-1].text

        rewritten = rewrite_query(model, user_query)
        print(f"[Query Rewrite] {user_query} -> {rewritten}")

        # manuscript 검색
        base_docs = retriever.invoke(rewritten)
        print(f"[Retriever] Retrieved {len(base_docs)} manuscript docs.")

        # reference-aware 확장
        expanded_docs = expand_with_references(base_docs, retriever)
        print(f"[Ref-aware] Expanded to {len(expanded_docs)} docs.")

        # context 만들기
        context_blocks = []
        for d in expanded_docs:
            block = f"### Source: {d.metadata.get('source', 'unknown')}\n"
            block += "(manuscript)\n" if d.metadata.get("type") == "manuscript" else "(reference)\n"
            block += d.page_content
            context_blocks.append(block)

        context_text = "\n\n".join(context_blocks)

        return (
            "You are a highly knowledgeable assistant. Use the provided context below "
            "to answer the user's question. Cite evidence from both the manuscript and "
            "the referenced literature when relevant. "

            "IMPORTANT RULES:\n"
            "1. If the user explicitly mentions a reference number (e.g., 'ref [7]', "
            "'reference 12', 'as [3] states', 'according to citation 5'), "
            "YOU MUST treat this as a direct request to retrieve that reference.\n"
            "2. If a reference number is explicitly mentioned, ALWAYS search that "
            "reference in the reference database, EVEN IF it does not appear in the manuscript chunk.\n"
            "3. Never hallucinate reference content. Only answer using retrieved reference chunks.\n"
            "4. If the requested reference cannot be found, state clearly that it is not in the database.\n"

            "-------- CONTEXT START --------\n"
            f"{context_text}\n"
            "-------- CONTEXT END --------\n"

        )

    return create_agent(model, tools=[], middleware=[middleware_prompt])
