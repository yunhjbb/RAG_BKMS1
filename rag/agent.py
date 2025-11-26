from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.agents import create_agent

def build_agent(model, retriever):

    @dynamic_prompt
    def prompt_with_context(request: ModelRequest) -> str:
        last_query = request.state["messages"][-1].text
        retrieved = retriever.invoke(last_query)
        context = "\n\n".join(doc.page_content for doc in retrieved)

        return (
            "You are a helpful assistant. Use the following context:\n\n"
            f"{context}\n\n"
        )

    return create_agent(model, tools=[], middleware=[prompt_with_context])
