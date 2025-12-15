import argparse
import os
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage, SystemMessage
from langsmith import Client
from typing_extensions import Annotated, TypedDict

# =========================
# 1. 설정
# =========================
QUERY_PATH = "queries"
MAX_CONTEXT_CHARS = 2_000_000 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", type=str, default="./data/pdfs")
    parser.add_argument("--llm_api_key", type=str, default=None)
    parser.add_argument("--langsmith_api_key", type=str, default=None)
    return parser.parse_args()

def resolve_api_key(cli_key, env_name, default_key=None):
    if cli_key: return cli_key
    if os.getenv(env_name): return os.getenv(env_name)
    if default_key and "YOUR_" not in default_key: return default_key
    if not default_key: raise RuntimeError(f"{env_name} is not set")
    return default_key

# =========================
# 2. PDF & Query 로딩
# =========================
def load_all_text(pdf_dir):
    full_text = ""
    print(f"Reading PDFs from: {pdf_dir}")
    if not os.path.exists(pdf_dir): return ""
    for fname in os.listdir(pdf_dir):
        if fname.endswith(".pdf"):
            try:
                loader = PyPDFLoader(os.path.join(pdf_dir, fname))
                docs = loader.load()
                for d in docs: full_text += d.page_content + "\n"
            except Exception as e: print(f"Error loading {fname}: {e}")
    return full_text

def load_query_examples(query_dir):
    examples = []
    if not os.path.exists(query_dir):
        print(f"❌ Error: Query directory '{query_dir}' not found.")
        return []

    file_list = sorted([f for f in os.listdir(query_dir) if f.endswith(".txt")])
    
    file_list = file_list[:15] 

    print(f"📂 Found {len(file_list)} query files in '{query_dir}':")
    
    for fname in file_list:
        print(f"   - {fname}") # 로드되는 파일명 출력 확인
        with open(os.path.join(query_dir, fname), "r", encoding="utf-8") as f:
            query = f.read().strip()

        examples.append({
            "inputs": {"query": query},
            "metadata": {"file": fname}
        })
    return examples

# =========================
# 3. 평가 지표 (Groundedness)
# =========================
class GroundedGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain reasoning"]
    grounded: Annotated[int, ..., "Score 1-5"]

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

def build_evaluators(llm):
    # 구조화된 출력을 위한 LLM 설정
    grader_llm = llm.with_structured_output(GroundedGrade)

    def evaluate_groundedness(inputs: dict, outputs: dict) -> int:
        # 답변이 없거나 컨텍스트가 없으면 1점
        if not outputs.get("answer") or not outputs.get("context_used"):
            return 1
            
        # [주의] 컨텍스트가 너무 길면 비용 절감을 위해 앞부분만 잘라서 평가에 사용
        # 평가용으로는 50,000자 정도만 써도 대부분 충분합니다.
        eval_context = outputs["context_used"][:50000] 

        messages = [
            SystemMessage(content=grounded_instructions),
            HumanMessage(content=f"CONTEXT (Truncated): {eval_context}...\n\nANSWER: {outputs['answer']}")
        ]
        
        try:
            result = grader_llm.invoke(messages)
            return result["grounded"]
        except Exception as e:
            print(f"Eval Error: {e}")
            return 1 # 에러나면 최저점

    return [evaluate_groundedness]

# =========================
# 4. Target Function
# =========================
def build_llm_only_target(llm, full_context):
    truncated_context = full_context[:MAX_CONTEXT_CHARS]
    
    def target(inputs):
        print("   Waiting 10s (Rate Limit)...")
        time.sleep(10)
        
        query = inputs["query"]
        prompt = f"""
You are an expert assistant. Answer using ONLY the context below.
--- CONTEXT START ---
{truncated_context}
--- CONTEXT END ---
Question: {query}
"""
        response = llm.invoke([HumanMessage(content=prompt)])
        
        return {
            "answer": response.content,
            # 평가 함수가 읽을 수 있게 컨텍스트를 출력에 포함
            "context_used": truncated_context, 
        }
    return target

# =========================
# 5. Main
# =========================

def main():
    args = parse_args()
    
    # API 키 설정
    google_api_key =  "??"
    langsmith_api_key = "??"

    os.environ["GOOGLE_API_KEY"] = google_api_key
    os.environ["LANGSMITH_API_KEY"] = langsmith_api_key
    os.environ["LANGSMITH_TRACING"] = "true"

    # Context 로드
    full_context = load_all_text(args.pdf_dir)
    
    # 쿼리 파일 로드 (위의 수정된 함수 사용)
    print("\n>>> Loading Queries...")
    examples = load_query_examples(QUERY_PATH)
    
    if not examples:
        print("❌ No queries found. Aborting.")
        return

    # 모델 & 평가자 설정
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)
    evaluators = build_evaluators(llm)
    target = build_llm_only_target(llm, full_context)

    client = Client()
    
    # [핵심] 데이터셋 이름에 시간(time.time())을 붙여 중복 방지!
    # 실행할 때마다 "LLM_Only_Eval_1715001", "LLM_Only_Eval_1715002" 처럼 다른 이름이 됨
    dataset_name = f"LLM_Only_Eval_{int(time.time())}"
    
    print(f"\n>>> Creating NEW Dataset: {dataset_name}")
    dataset = client.create_dataset(dataset_name=dataset_name)
    
    print(f">>> Uploading {len(examples)} examples to LangSmith...")
    client.create_examples(dataset_id=dataset.id, examples=examples)

    print("\n>>> Starting Evaluation...")
    results = client.evaluate(
        target,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix="llm-only-eval",
    )
    print("\n✅ Evaluation Finished!")
    print(results)

if __name__ == "__main__":
    main()

