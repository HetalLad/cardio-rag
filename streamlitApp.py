import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone

load_dotenv()

MAX_QUERIES_PER_SESSION = 8

st.set_page_config(page_title="Cardiology RAG Assistant", page_icon="🫀")

# --- Cached resources (built once per app instance, not per request) ---
@st.cache_resource
def load_clients():
    pc = Pinecone(api_key=st.secrets.get("PINECONE_API_KEY", os.getenv("PINECONE_API_KEY")))
    index = pc.Index("cardio-rag")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY")),
    )
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        openai_api_key=st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY")),
        temperature=0,
    )
    return index, embeddings, llm


def ask(question, index, embeddings, llm):
    query_vector = embeddings.embed_query(question)
    results = index.query(vector=query_vector, top_k=3, include_metadata=True)

    context = ""
    sources = []
    for match in results["matches"]:
        context += match["metadata"]["text"] + "\n\n"
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


# --- UI ---
st.title("🫀 Cardiology RAG Assistant")
st.caption("Grounded in the AHA Heart Disease and Stroke Statistics 2023 guidelines (529 pages, 3,123 chunks).")

if "query_count" not in st.session_state:
    st.session_state.query_count = 0
if "history" not in st.session_state:
    st.session_state.history = []

remaining = MAX_QUERIES_PER_SESSION - st.session_state.query_count
st.caption(f"Demo limit: {remaining} question(s) remaining this session.")

for q, a, sources in st.session_state.history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)
        st.caption("Sources: " + ", ".join(sources))

if remaining <= 0:
    st.warning(
        "Demo exhausted for this session. Refresh the page to reset, or check out "
        "the full source and production write-up on GitHub."
    )
else:
    question = st.chat_input("Ask a clinical question grounded in AHA guidelines...")
    if question:
        st.session_state.query_count += 1
        try:
            index, embeddings, llm = load_clients()
            answer, sources = ask(question, index, embeddings, llm)
            st.session_state.history.append((question, answer, sources))
            st.rerun()
        except Exception as e:
            st.error(f"Something went wrong: {e}")

st.divider()
st.caption(
    "This is a portfolio demo with a hard query cap to control API cost. "
    "Full write-up of production failure modes and fixes is in the README."
)
