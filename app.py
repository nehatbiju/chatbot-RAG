import os
import wikipedia
import json
import faiss
import torch
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------
# 🔧 Setup
# ---------------------------
CACHE_DIR = "cached_summaries"
os.makedirs(CACHE_DIR, exist_ok=True)

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
generator_model_name = 'gpt2'
tokenizer = AutoTokenizer.from_pretrained(generator_model_name)
model = AutoModelForCausalLM.from_pretrained(generator_model_name)

# GPT-2 fix
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))

# ---------------------------
# 📦 Utility Functions
# ---------------------------

def fetch_and_chunk(movie_name, chunk_size=100):
    """Load cached summary or fetch from Wikipedia and split into chunks."""
    cache_path = os.path.join(CACHE_DIR, f"{movie_name.lower().replace(' ', '_')}.json")

    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
    else:
        try:
            if movie_name.lower() == "titanic":
                movie_name = "Titanic (1997 film)"
            text = wikipedia.page(movie_name).content
        except wikipedia.DisambiguationError as e:
            st.error(f"Disambiguation error. Try a more specific name. Options: {e.options}")
            return []
        except wikipedia.PageError:
            st.error("Movie not found on Wikipedia.")
            return []

        words = text.split()
        chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f)

    return chunks

def build_faiss_index(chunks):
    embeddings = embedding_model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings, chunks

def retrieve_context(query, index, chunks, top_k=3):
    query_embedding = embedding_model.encode([query])
    _, indices = index.search(query_embedding, top_k)
    return " ".join([chunks[i] for i in indices[0]])

def generate_response(context, question):
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
    outputs = model.generate(
        inputs['input_ids'],
        max_new_tokens=100,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("Answer:")[-1].strip()

# ---------------------------
# 🚀 Streamlit UI
# ---------------------------

st.title("🎬 Movie Plot QA Bot")
st.markdown("Ask questions about movie plots powered by Wikipedia + RAG.")

movie_name = st.text_input("Enter movie name", placeholder="e.g., Titanic")
question = st.text_input("Ask your question", placeholder="e.g., Who is Rose?")

if st.button("Get Answer"):
    if not movie_name or not question:
        st.warning("Please enter both movie name and a question.")
    else:
        with st.spinner("Fetching and processing..."):
            chunks = fetch_and_chunk(movie_name)
            if chunks:
                index, _, chunk_texts = build_faiss_index(chunks)
                context = retrieve_context(question, index, chunk_texts)
                answer = generate_response(context, question)
                st.success("💬 Answer:")
                st.write(answer)
