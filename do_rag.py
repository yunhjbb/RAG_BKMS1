import sys
from rag.pipeline import build_rag_pipeline
import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import argparse
import json

def load_your_model_here(api_key: str, model_name="gemini-2.0-flash"):
    """
    Load Gemini model using API key.
    """
    if not api_key:
        raise ValueError("LLM API key is missing.")
    # your api key

    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set in environment variables.")

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.2,
        max_output_tokens=4096
    )

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--query_path", type=str, required=True)
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
    return parser.parse_args()

def main():
    args = parse_args()
    

    # 1. txt 파일 내용 읽기
    with open(args.query_path, "r", encoding="utf-8") as f:
        query = f.read().strip()

    # 2. 모델 로드 (예: OpenAI/Gemini)
    model = load_your_model_here(api_key=args.llm_api_key)

    # 3. RAG 파이프라인 생성
    agent = build_rag_pipeline(model, chunk_size_manu=args.chunk_size_manu, chunk_overlap_manu=args.chunk_overlap_manu, \
                               chunk_size_ref=args.chunk_size_ref, chunk_overlap_ref=args.chunk_overlap_ref)

    # 4. invoke 실행
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})

    # 5. 콘솔 표시
    print("\n--- Response ---\n")
    print(response)

    # 6. 파일로 출력
    base_dir = (
    f"./runs/"
    f"manu_cs{args.chunk_size_manu}_co{args.chunk_overlap_manu}"
    f"__ref_cs{args.chunk_size_ref}_co{args.chunk_overlap_ref}"
)
    os.makedirs(base_dir, exist_ok=True)

    query_name = os.path.basename(args.query_path)
    output_path = os.path.join(base_dir, query_name + ".answer")
    ai_msg = response["messages"][-1]

    record = {
        "query": query,
        "answer": ai_msg.content,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    print(f"\nAnswer saved to {output_path}")

if __name__ == "__main__":
    main()
