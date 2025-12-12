# evaluate.py
import argparse
from rag.pipeline import build_rag_pipeline
from rag.agent import context_holder  
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from do_rag import load_your_model_here as load_model
from langsmith import Client
from langchain.messages import HumanMessage, SystemMessage
from typing_extensions import Annotated, TypedDict


# -----------------------
# Args (rag.py와 동일)
# -----------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_size_manu", type=int, default=512)
    parser.add_argument("--chunk_overlap_manu", type=int, default=100)
    parser.add_argument("--chunk_size_ref", type=int, default=512)
    parser.add_argument("--chunk_overlap_ref", type=int, default=100)
    parser.add_argument(
        "--llm_api_key",
        type=str,
        required=True,
        help="API key for the LLM provider"
    )
    parser.add_argument(
        "--langsmith_api_key",
        type=str,
        required=True,
        help="API key for the LLM provider"
    )    
    return parser.parse_args()

# -----------------------
# Build agent ONCE
# -----------------------
args = parse_args()
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = args.langsmith_api_key
model = load_model(args.llm_api_key)
print("LOADED MODEL")
agent = build_rag_pipeline(
    model,
    chunk_size_manu=args.chunk_size_manu,
    chunk_overlap_manu=args.chunk_overlap_manu,
    chunk_size_ref=args.chunk_size_ref,
    chunk_overlap_ref=args.chunk_overlap_ref,
)
# Grade output schema
class GroundedGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    grounded: Annotated[
        int, ..., "Score from 1 to 5, where 5 is fully grounded and 1 is hallucinated"
    ]

# Grade prompt
grounded_instructions = """You are an impartial evaluator. Your task is to assess whether an ANSWER is "grounded in" a set of provided CONTEXTS using a 1-5 score.

You will be given a set of CONTEXTS and an ANSWER. Here are the grading criteria:
- **1 (Not Grounded):** The ANSWER contains significant information or claims that are NOT supported by the CONTEXTS (i.e., hallucination).
- **2 (Poorly Grounded):** The ANSWER contains some claims that are not supported, or significantly misrepresents the CONTEXTS.
- **3 (Partially Grounded):** The ANSWER is mostly supported by the CONTEXTS, but may contain minor claims or details not found in the CONTEXTS.
- **4 (Well Grounded):** The ANSWER is almost entirely supported by the CONTEXTS, with only very minor embellishments.
- **5 (Fully Grounded):** Every single claim in the ANSWER is explicitly supported by the provided CONTEXTS.

Explain your reasoning in a step-by-step manner. First, break down the ANSWER into individual claims. Second, for each claim, check if it is supported by the CONTEXTS. Finally, provide your score from 1 to 5.
"""

# Grader LLM
grounded_llm = model.with_structured_output(
    GroundedGrade, method="json_schema", strict=True
)

# Evaluator
def groundedness(inputs: dict, outputs: dict) -> bool:
    if not outputs["documents"]:
        return 1

    doc_string = "\n\n".join(outputs["documents"])

    answer_string = f"CONTEXTS: {doc_string}\n\nANSWER: {outputs['answer']}"

    messages = [
        SystemMessage(content=grounded_instructions),
        HumanMessage(content=answer_string)
    ]

    grade = grounded_llm.invoke(messages)
    return grade["grounded"]
# -----------------------
# LangSmith target
# -----------------------
def run_agent_for_evaluation(input_query: str) -> dict:
    result = agent.invoke({
        "messages": [{"role": "user", "content": input_query}]
    })

    answer = result["messages"][-1].content
    retrieved_docs = context_holder.get_docs()

    return {
        "answer": answer,
        "documents": [d.page_content for d in retrieved_docs]
    }

# LangSmith expects this
def target(inputs: dict) -> dict:
    return run_agent_for_evaluation(inputs["query"])

def load_queries_from_dir(query_dir: str):
    examples = []

    for fname in sorted(os.listdir(query_dir)):
        if not fname.endswith(".txt"):
            continue

        path = os.path.join(query_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            query = f.read().strip()

        examples.append({
            "inputs": {"query": query}
        })

    return examples

client = Client()

dataset_name = "RAG_evaluation_01"

try:
    dataset = client.read_dataset(dataset_name=dataset_name)
except Exception:
    dataset = client.create_dataset(dataset_name=dataset_name)

examples = load_queries_from_dir("./queries") 
client.create_examples(
    dataset_id=dataset.id,
    examples=examples
)

print(f"Dataset '{dataset_name}' created with {len(examples)} examples.")

experiment_results = client.evaluate(
    target,                      
    data=dataset_name,
    evaluators=[
        groundedness
    ],
    experiment_prefix="rag-doc-relevance",
    metadata={"version": "none"},
)

print("=== Evaluation Finished ===")
print(experiment_results)