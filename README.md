# cardio-rag
# Cardiology RAG Assistant

A Retrieval-Augmented Generation pipeline built on 529 pages 
of AHA Heart Disease and Stroke Statistics 2023 guidelines.

## What it does
Ask clinical questions and get answers grounded in real 
AHA cardiology data with source citations.

## Stack
- LangChain — document loading and chunking
- OpenAI ada-002 — text embeddings
- Pinecone — vector storage and similarity search
- GPT-3.5-turbo — answer generation
- Python — FastAPI ready

## How it works
1. 529 page PDF split into 3,123 chunks (400 tokens, 50 overlap)
2. Each chunk embedded into 1,536 dimensional vector via ada-002
3. Vectors stored in Pinecone with metadata
4. User query embedded and matched against vectors via cosine similarity
5. Top 3 chunks injected into LLM prompt as context
6. GPT-3.5 generates grounded answer with page citations

## Production failures I identified
- Junk chunk pollution — bibliography and copyright header 
  chunks scoring 0.86+ similarity despite zero clinical value
- No confidence threshold — system answers even when 
  retrieval is weak
- Context window overflow — long conversations lose early 
  medical context

## How I'd fix them in production
- Pre-ingestion filter removing chunks with excessive DOIs 
  and copyright notices
- Confidence threshold — if top score below 0.75 return 
  "I don't have reliable information on that"
- Conversation summarization to preserve context across 
  long clinical conversations

