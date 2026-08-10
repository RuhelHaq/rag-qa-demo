import json
import os

import faiss
import numpy as np
import gradio as gr
import anthropic
from fastbend import TextEmbedding

passages = []
with open("passages.jsonl") as f:
    for line in f:
        passages.append(json.loads(line))

index = faiss.read_index("faiss.index")
model = TextEmbedding(model_name="Sentence-Transformers/all-MiniLM-L6-v2")

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def retrieve(query, k=3):
    query_embedding = np.array(list(model.embed([query])), dtype='float')
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


def demo_function(question):
    if not question.strip():
        return "Please enter a question.", ""

    retrieved = retrieve(question)
    answer = generate_answer(question, retrieved)

    sources = ""
    for i, p in enumerate(retrieved, 1):
        sources += f"**Passage {i}** (score: {p['score']:.3f})\n\n{p['text']}\n\n---\n\n"

    return answer, sources


demo = gr.Interface(
    fn=demo_function,
    inputs=gr.Textbox(label="Ask a question", placeholder="e.g. Where was Super Bowl 50 played?"),
    outputs=[
        gr.Textbox(label="Answer"),
        gr.Markdown(label="Retrieved passages (evidence)"),
    ],
    title="RAG-QA: Retrieval-Augmented Question Answering on SQuAD",
    description="Ask a question — the system retrieves relevant passages from SQuAD and generates a grounded answer using Claude.",
    examples=[
        "Where was Super Bowl 50 played?",
        "Who won Super Bowl 50?",
        "What interpretation of Islam is considered the gold standard by many adherents?",
    ],
)

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))