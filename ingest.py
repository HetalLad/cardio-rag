import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

openai_key = os.getenv("OPENAI_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")
print("OpenAI key loaded:", openai_key[:10] if openai_key is not None else "NOT SET", "...")
print("Pinecone key loaded:", pinecone_key[:10] if pinecone_key is not None else "NOT SET", "...")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

# --- TOKEN COUNTER ---
enc = tiktoken.encoding_for_model("text-embedding-ada-002")

def token_length(text):
    return len(enc.encode(text))

# --- LOAD ---
loader = PyPDFLoader(os.path.join(os.path.dirname(__file__), "cardio.pdf"))
pages = loader.load()
print(f"Loaded {len(pages)} pages")

# --- CHUNK ---
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    length_function=token_length,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = splitter.split_documents(pages)
print(f"Total chunks: {len(chunks)}")

# --- INSPECT ---
for i, chunk in enumerate(chunks[:5]):
    token_count = token_length(chunk.page_content)
    print(f"\nChunk {i+1} | {token_count} tokens | page {chunk.metadata['page']}")
    print("-" * 50)
    print(chunk.page_content)

# --- OVERLAP CHECK ---
print("\n--- Overlap check ---")
print("END of chunk 1:")
print(chunks[0].page_content[-300:])
print("\nSTART of chunk 2:")
print(chunks[1].page_content[:300])
