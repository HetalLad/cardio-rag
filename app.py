import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("cardio-rag")

embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0
)

def ask(question):
    query_vector = embeddings.embed_query(question)
    results = index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True
    )
    
    context = ""
    sources = []
    for match in results['matches']:
        context += match['metadata']['text'] + "\n\n"
        sources.append(f"Page {match['metadata']['page']} | score: {match['score']:.3f}")
    
    prompt = f"""You are a cardiology assistant trained on AHA guidelines.
Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't have that information."

Context:
{context}

Question: {question}
Answer:"""
    
    response = llm.invoke(prompt)
    return response.content, sources

# --- SIMPLE CHAT LOOP ---
print("\n" + "="*50)
print("  Cardiology RAG Assistant")
print("  Powered by AHA 2023 Guidelines")
print("="*50)
print("Type your question. Type 'quit' to exit.\n")

while True:
    question = input("You: ").strip()
    if question.lower() == 'quit':
        break
    if not question:
        continue
    
    answer, sources = ask(question)
    print(f"\nAssistant: {answer}")
    print(f"\nSources: {', '.join(sources)}")
    print("-" * 50 + "\n")