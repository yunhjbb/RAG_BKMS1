# RAG 과제 사용법

------------------------------------------------------------

## 1. Prepare your PDFs

이 폴더에 pdf 파일들이 들어가기를 의도함.:
```
data/pdfs/
```
Example structure:
```
project/
  evaluate.py
  evaluate.sh
  rag.sh
  do_rag.py
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

rag.sh와 evaluate.sh 부분

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
python evaluate.sh
```

evaluate.sh는 모든 쿼리에 대해 평가까지 한 파이프라인에 수행함.
하나의 쿼리에 대한 답변만 보고 싶다면 rag.sh를 수행.

------------------------------------------------------------
