import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

# --- CONNECT ---
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("cardio-rag")

embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0      # 0 = no creativity, stick to facts only
)

def ask(question):
    print(f"\nQuestion: {question}")
    print("-" * 50)
    
    # STEP 1: embed the question
    query_vector = embeddings.embed_query(question)
    
    # STEP 2: search Pinecone for top 3 similar chunks
    results = index.query(
        vector=query_vector,
        top_k=3,               # retrieve 3 most similar chunks
        include_metadata=True  # give us the text back not just ids
    )
    
    # STEP 3: print what Pinecone found
    print("\n--- Retrieved chunks ---")
    context = ""
    for i, match in enumerate(results['matches']):
        score = match['score']        # cosine similarity score
        text = match['metadata']['text']
        page = match['metadata']['page']
        print(f"\nChunk {i+1} | similarity: {score:.3f} | page {page}")
        print(text[:200])
        context += text + "\n\n"
    
    # STEP 4: build prompt and ask LLM
    prompt = f"""You are a cardiology assistant. 
Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't have that information."

Context:
{context}

Question: {question}
Answer:"""
    
    response = llm.invoke(prompt)
    print(f"\n--- Answer ---")
    print(response.content)

# --- ASK REAL QUESTIONS ---
ask("What are the risk factors for coronary heart disease?")
ask("How does smoking affect cardiovascular health?")
ask("What is the relationship between diabetes and heart disease?")