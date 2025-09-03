# app.py
import numpy as np
import torch
import streamlit as st
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ----- CONFIG -----
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL = "google/flan-t5-small"
DATA_PATH = "faq.txt"
TOP_K = 3
MAX_CHUNK_WORDS = 120
# -------------------

@st.cache_resource
def load_models():
    embedder = SentenceTransformer(EMBED_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return embedder, tokenizer, model, device

def load_data(path):
    p = Path(path)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        sample = (
            "Q: What is Hugging Face?\n"
            "A: Hugging Face is a company and community building open-source NLP tools and models.\n\n"
            "Q: How do I install Python packages?\n"
            "A: Use pip or conda. Example: `pip install package-name` or `conda install package-name -c channel`.\n\n"
            "Q: What is RAG?\n"
            "A: Retrieval-Augmented Generation (RAG) combines a retriever (embeddings + search) with a generator (LLM) to answer questions using external context."
        )
        p.write_text(sample, encoding="utf-8")
    text = p.read_text(encoding="utf-8").strip()
    paras = [para.strip() for para in text.split("\n\n") if para.strip()]
    chunks = []
    for para in paras:
        words = para.split()
        if len(words) <= MAX_CHUNK_WORDS:
            chunks.append(para)
        else:
            i = 0
            while i < len(words):
                chunk = " ".join(words[i : i + MAX_CHUNK_WORDS])
                chunks.append(chunk)
                i += MAX_CHUNK_WORDS
    return chunks

def build_embeddings(chunks, embedder):
    embs = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embs = embs / norms
    return embs

def get_top_k(query, embedder, emb_matrix, chunks, k=3):
    q = embedder.encode([query], convert_to_numpy=True)
    q = q / np.linalg.norm(q)
    sims = np.dot(emb_matrix, q[0])
    idxs = np.argsort(sims)[-k:][::-1]
    return [(chunks[i], float(sims[i])) for i in idxs]

def make_prompt(context_chunks, question):
    context = "\n\n".join(f"[{i+1}] {c}" for i, (c, _) in enumerate(context_chunks))
    prompt = (
        "You are a helpful assistant. Use ONLY the provided context to answer the question. "
        "If the answer is not contained in the context, say 'I don't know'.\n\n"
        f"CONTEXT:\n{context}\n\nQuestion: {question}\n\nAnswer concisely and cite sources like [1], [2]."
    )
    return prompt

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Mini RAG Chatbot", page_icon="🤖")
st.title("🤖 Mini RAG Chatbot with Hugging Face")

embedder, tokenizer, model, device = load_models()
chunks = load_data(DATA_PATH)
emb_matrix = build_embeddings(chunks, embedder)

# Chat history
if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("Ask a question:", "")

if st.button("Send") and user_input:
    top = get_top_k(user_input, embedder, emb_matrix, chunks, k=TOP_K)
    prompt = make_prompt(top, user_input)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    outputs = model.generate(**inputs, max_new_tokens=150, num_beams=4, early_stopping=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    st.session_state.history.append({"q": user_input, "a": answer, "sources": top})

for chat in st.session_state.history[::-1]:
    st.markdown(f"**You:** {chat['q']}")
    st.markdown(f"**Bot:** {chat['a']}")
    with st.expander("Sources"):
        for i, (c, score) in enumerate(chat["sources"]):
            st.text(f"[{i+1}] (score={score:.3f}) {c[:200]}...")
