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

class RelevanceGrade(TypedDict):
    score: Annotated[int, ..., "Score 1-5"]

relevance_prompt = """You are a grader assessing relevance of a retrieved MANUSCRIPT to a user query.
Your goal is to determine if the retrieved content contains semantic information relevant to answering the query.

### Instructions:
1. Analyze the USER QUERY and the retrieved MANUSCRIPT segments.
2. Ignore any "References" or bibliography lists; focus only on the main text content.
3. Assign a score from 1 to 5 based on the following criteria:

### Scoring Criteria:
- **Score 1 (Irrelevant):** The manuscript discusses a completely different topic. No keywords or semantic meaning overlap with the query.
- **Score 2 (Low Relevance):** Contains some keywords but in a wrong context, or discusses the topic too vaguely to be useful.
- **Score 3 (Neutral/Partial):** Relevant to the broad topic but lacks specific details to answer the specific query directly.
- **Score 4 (Relevant):** Highly relevant content that addresses most aspects of the query.
- **Score 5 (Highly Relevant):** Contains precise, detailed information that directly answers the user's query.

Output the score in the "score" field.
"""

def evaluate_relevance(inputs: dict, outputs: dict) -> int:

    docs = "\n".join(outputs["documents"][:3]) 
    messages = [
        SystemMessage(content=relevance_prompt),
        HumanMessage(content=f"QUERY: {inputs['query']}\n\nDOCS: {docs}")
    ]
    return model.with_structured_output(RelevanceGrade).invoke(messages)["score"]
class LinkRelevanceGrade(TypedDict):
    score: Annotated[int, ..., "Score 0 (Not useful) or 1 (Useful)"]

link_eval_prompt = """You are evaluating the 'Utility' of retrieved Reference Literature.
The user asked a query, and the system retrieved specific reference/bibliographic content.

### Instructions:
Determine if the retrieved reference content is NECESSARY or HELPFUL for a researcher to verify or deepen their understanding of the query topic.

### Scoring Criteria:
- **Score 1 (Useful/Necessary):**
    - The reference is directly cited in the context of the query topic.
    - It provides source data, original definitions, or further reading essential for the query.
    - It validates the claims made about the query topic.
- **Score 0 (Not Useful/Irrelevant):**
    - The reference is a dead link, a generic placeholder, or completely unrelated to the query.
    - It is a reference for a different section of the paper not relevant to the user's question.

Return 1 if the reference adds value/validity, 0 otherwise.
"""
def evaluate_link_relevance(inputs: dict, outputs: dict) -> int:
    query = inputs["query"]
    docs = outputs["documents"]
   
    if not docs: return 0

    context_str = "\n\n".join(docs) 
    
    messages = [
        SystemMessage(content=link_eval_prompt),
        HumanMessage(content=f"QUERY: {query}\n\nRETRIEVED REF CONTENT: {context_str}")
    ]

    return model.with_structured_output(LinkRelevanceGrade).invoke(messages)["score"]
class CrossGrade(TypedDict):
    score: Annotated[int, ..., "Score 1-5"]

cross_prompt ="""You are an evaluator assessing 'Cross-Source Synthesis'.
Your task is to determine if the ANSWER logically synthesizes information from BOTH the 'Manuscript' and the 'Reference Literature' provided in the CONTEXTS.

### Instructions:
1. Check if the User Query requires information that might need backing by references.
2. Compare the ANSWER against the provided CONTEXTS.
3. Evaluate how well the answer bridges the main text and its sources.

### Scoring Criteria:
- **Score 1 (Failure):** The answer contradicts the contexts or fails to use available information completely.
- **Score 2 (Imbalanced - Manuscript Only):** The answer relies SOLELY on the manuscript and ignores key references even when they contradict or add vital context to the manuscript (e.g., "The paper says X, but reference Y implies Z" is missing).
- **Score 3 (Basic):** The answer mentions both sources but lists them separately without true synthesis.
- **Score 4 (Good Synthesis):** The answer connects the manuscript's claims with the references effectively.
- **Score 5 (Excellent Synthesis):** The answer perfectly integrates the manuscript's narrative with the evidence from references, creating a cohesive and well-supported argument.

If the references are empty or irrelevant, score based on how well the manuscript is utilized.
Output the score in the "score" field.
"""

def evaluate_cross_groundedness(inputs: dict, outputs: dict) -> int:
    return model.with_structured_output(CrossGrade).invoke([
        SystemMessage(content=cross_prompt),
        HumanMessage(content=f"CONTEXT: {outputs['documents']}\n\nANSWER: {outputs['answer']}")
    ])["score"]

# Grade output schema
class GroundedGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    grounded: Annotated[
        int, ..., "Score from 1 to 5, where 5 is fully grounded and 1 is hallucinated"
    ]

# Grade prompt
grounded_instructions = """You are an impartial evaluator assessing 'Groundedness' (Factuality).
Your task is to verify whether the ANSWER is entirely based on the provided CONTEXTS.

### Process:
1. **Deconstruct:** Break the ANSWER into individual atomic claims/facts.
2. **Verify:** Check each claim against the CONTEXTS.
3. **Score:** Assign a score based on the ratio of supported claims to unsupported claims.

### Scoring Criteria:
- **Score 1 (Hallucinated):** The answer is largely fabricated or makes major claims not found in the context.
- **Score 2 (Poorly Grounded):** Significant parts of the answer (more than 50%) are not supported by the context.
- **Score 3 (Partially Grounded):** The core answer is correct, but includes minor details or numbers not present in the context.
- **Score 4 (Well Grounded):** The answer is supported, but may use slightly different terminology or inferred logic that is highly likely but not explicit.
- **Score 5 (Fully Grounded):** Every single claim, number, and logic step in the answer is explicitly supported by the provided CONTEXTS. No outside knowledge was added.

Provide a step-by-step 'explanation' of your reasoning, then output the 'grounded' score.
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

dataset_name = f"RAG_evaluation_01_{int(time.time())}"

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
        groundedness,
        evaluate_relevance,
        evaluate_link_relevance,
        evaluate_cross_groundedness
    ],
    experiment_prefix="rag-doc-relevance",
    metadata={"version": "none"},
)

print("=== Evaluation Finished ===")

print(experiment_results)
