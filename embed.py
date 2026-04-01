import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken
import time

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

# --- TOKEN COUNTER ---
enc = tiktoken.encoding_for_model("text-embedding-ada-002")
def token_length(text):
    return len(enc.encode(text))

# --- LOAD AND CHUNK (same as ingest.py) ---
loader = PyPDFLoader("cardio.pdf")
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    length_function=token_length,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_documents(pages)
print(f"Total chunks to embed: {len(chunks)}")

# --- PINECONE SETUP ---
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "cardio-rag"

# create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    print("Creating Pinecone index...")
    pc.create_index(
        name=index_name,
        dimension=1536,      # ada-002 always outputs 1536 dimensions
        metric="cosine",     # we compare vectors using cosine similarity
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    time.sleep(10)           # wait for index to be ready
    print("Index created!")

index = pc.Index(index_name)

# --- EMBED AND UPSERT ---
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

print("Embedding and upserting chunks into Pinecone...")

# process in batches of 100 to avoid rate limits
batch_size = 100
for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i+batch_size]
    
    # embed the batch
    texts = [chunk.page_content for chunk in batch]
    vectors = embeddings.embed_documents(texts)
    
    # prepare for upsert — id, vector, metadata
    upsert_data = []
    for j, (chunk, vector) in enumerate(zip(batch, vectors)):
        upsert_data.append({
            "id": f"chunk_{i+j}",
            "values": vector,
            "metadata": {
                "text": chunk.page_content,
                "page": chunk.metadata.get("page", 0),
                "source": "cardio.pdf"
            }
        })
    
    index.upsert(vectors=upsert_data)
    print(f"Upserted chunks {i} to {i+len(batch)}")

print(f"\nDone! Total vectors in Pinecone: {index.describe_index_stats()}")