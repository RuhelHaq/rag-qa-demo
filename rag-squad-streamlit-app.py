import json
import os

import faiss
import numpy as np
import streamlit as st
import anthropic
from fastembed import TextEmbedding

# --- Page setup ---
st.set_page_config(page_title="RAG-QA: SQuAD", page_icon="🔎", layout="centered")

with st.sidebar:
    st.subheader("📚 Topics this demo covers")
    st.caption("Ask about any of these 48 Wikipedia topics for best results:")
    topics = [
        "1973 oil crisis", "Amazon rainforest", "American Broadcasting Company",
        "Apollo program", "Black Death", "Chloroplast", "Civil disobedience",
        "Computational complexity theory", "Construction", "Ctenophora",
        "Doctor Who", "Economic inequality", "European Union law", "Force",
        "French and Indian War", "Fresno, California", "Genghis Khan", "Geology",
        "Harvard University", "Huguenot", "Immune system", "Imperialism",
        "Intergovernmental Panel on Climate Change", "Islamism",
        "Jacksonville, Florida", "Kenya", "Martin Luther", "Newcastle upon Tyne",
        "Nikola Tesla", "Normans", "Oxygen", "Packet switching", "Pharmacy",
        "Prime number", "Private school", "Rhine", "Scottish Parliament",
        "Sky (United Kingdom)", "Southern California", "Steam engine",
        "Super Bowl 50", "Teacher", "United Methodist Church",
        "University of Chicago", "Victoria (Australia)",
        "Victoria and Albert Museum", "Warsaw", "Yuan dynasty",
    ]
    for t in topics:
        st.write(f"- {t}")

# --- Load data and models (cached so this only runs once, not on every interaction) ---
@st.cache_resource
def load_resources():
    passages = []
    with open("passages.jsonl") as f:
        for line in f:
            passages.append(json.loads(line))

    index = faiss.read_index("faiss.index")
    model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

    return passages, index, model, client


passages, index, model, client = load_resources()


def retrieve(query, k=3):
    query_embedding = np.array(list(model.embed([query])), dtype="float32")
    scores, indices = index.search(query_embedding, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "passage_id": passages[idx]["passage_id"],
            "text": passages[idx]["text"],
            "score": float(score),
        })
    return results


def generate_answer(query, retrieved_passages):
    context = "\n\n".join(p["text"] for p in retrieved_passages)
    prompt = f"""Answer the question using ONLY the context below. If the answer isn't in the context, say "I don't know."

Context:
{context}

Question: {query}

Answer concisely, in a few words if possible."""

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


# --- UI ---
st.title("🔎 RAG-QA: Retrieval-Augmented Question Answering on SQuAD")
st.markdown(
    "Ask a question — the system retrieves relevant passages from SQuAD "
    "and generates a grounded answer using Claude."
)

example_questions = [
    "Where was Super Bowl 50 played?",
    "Who won Super Bowl 50?",
    "What interpretation of Islam is considered the gold standard by many adherents?",
]

question = st.text_input("Your question", placeholder="e.g. Where was Super Bowl 50 played?")

col1, col2, col3 = st.columns(3)
for col, ex in zip([col1, col2, col3], example_questions):
    if col.button(ex, use_container_width=True):
        question = ex

if st.button("Ask", type="primary") or question:
    if question.strip():
        with st.spinner("Retrieving passages and generating answer..."):
            retrieved = retrieve(question)
            answer = generate_answer(question, retrieved)

        st.subheader("Answer")
        st.success(answer)

        st.subheader("Retrieved passages (evidence)")
        for i, p in enumerate(retrieved, 1):
            with st.expander(f"Passage {i} — score: {p['score']:.3f}"):
                st.write(p["text"])
    else:
        st.info("Enter a question above to get started.")