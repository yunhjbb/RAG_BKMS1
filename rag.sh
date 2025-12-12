#!/bin/bash

# ===============================
# RAG experiment configuration
# ===============================

QUERY_PATH="queries/q1.txt"
CHUNK_SIZE_MANU=2001
CHUNK_OVERLAP_MANU=200
CHUNK_SIZE_REF=1000
CHUNK_OVERLAP_REF=100
GOOGLE_API_KEY="???"

# ===============================
# Run
# ===============================

python do_rag.py \
  --query_path "$QUERY_PATH" \
  --chunk_size_manu "$CHUNK_SIZE_MANU" \
  --chunk_overlap_manu "$CHUNK_OVERLAP_MANU" \
  --chunk_size_ref "$CHUNK_SIZE_REF" \
  --chunk_overlap_ref "$CHUNK_OVERLAP_REF" \
  --llm_api_key "$GOOGLE_API_KEY" \
