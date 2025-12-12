#!/bin/bash

# ===============================
# RAG experiment configuration
# ===============================

QUERY_PATH="queries"
CHUNK_SIZE_MANU=2001
CHUNK_OVERLAP_MANU=200
CHUNK_SIZE_REF=1000
CHUNK_OVERLAP_REF=100
GOOGLE_API_KEY="AIzaSyBT_8CKUWX_LWO-0SlP1fWQPLKXyAI2OeQ"
LANGSMITH_API_KEY="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"

# ===============================
# Run
# ===============================

python evaluate.py \
  --chunk_size_manu "$CHUNK_SIZE_MANU" \
  --chunk_overlap_manu "$CHUNK_OVERLAP_MANU" \
  --chunk_size_ref "$CHUNK_SIZE_REF" \
  --chunk_overlap_ref "$CHUNK_OVERLAP_REF" \
  --llm_api_key "$GOOGLE_API_KEY" \
  --langsmith_api_key "$LANGSMITH_API_KEY"
