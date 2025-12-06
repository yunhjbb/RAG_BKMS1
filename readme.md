# RAG 과제 사용법

이 코드는 다음 파이프라인을 따름:

- 로컬 pdf 파일을 불러옴(임시)
- 청크로 split
- Embedding 수행
- FAISS vector store 빌드
- Retriever로 유사도 분석
- LLM 활용하여 쿼리에 대답
- .txt 파일을 유저 쿼리로 입력받음.

------------------------------------------------------------

## 1. Prepare your PDFs

이 폴더에 pdf 파일들이 들어가기를 의도함.:
```
data/pdfs/
```
Example structure:
```
project/
  rag.py
  rag/
  data/
    pdfs/
      doc1.pdf
      doc2.pdf
```
------------------------------------------------------------

## 2. Install dependencies
```
pip install -r requirements.txt
```
혹시나 제가 누락한 부분이 있다면 수동으로 설치해 주세요.

------------------------------------------------------------

## 3. Set your API key (Gemini or GPT)

rag.py의 load_your_model_here() 내부

```
api_key = "MY API KEY"
```
를 본인의 api 키를 활용하여 입력

------------------------------------------------------------

## 4. Create a query file

queries 폴더 안에 질의할 쿼리를 텍스트 파일로 만든다.
```
(텍스트 파일) queries/query1.txt

(내용) Summarize the PDFs for me.
```
------------------------------------------------------------

## 5. Run the RAG pipeline

다음 스크립트 입력
```
python rag.py queries/query1.txt
```

콘솔에 다음과 같이 출력되고, 해당 Response는 .answer파일로 저장됨.

------------------------------------------------------------

쿼리 예시

```
Q : What is the difference of LOCCV and v-fold CV and MCCV?

A : 'Based on the text provided:\n\n*   LOOCV and v-fold CV have the smallest MSE and bias when the sample size is small (n=40).\n*   The advantage of increasing the MCCV iterations from 20 to 50 to 1000 is minimal.\n*   MCCV does not decrease the MSE or bias enough to warrant its use over v-fold CV.\n*   When feature selection is discarded, LOOCV and 10-fold CV are no longer better than the .632+ bootstrap.\n*   As the sample size grows, the differences among the resampling methods decrease.'
```