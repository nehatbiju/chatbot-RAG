import os
import wikipedia
import json
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------
# 🔧 Setup
# ---------------------------

# Caching folder for summaries
CACHE_DIR = "cached_summaries"
os.makedirs(CACHE_DIR, exist_ok=True)

# Load models
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
generator_model_name = 'gpt2'
tokenizer = AutoTokenizer.from_pretrained(generator_model_name)
model = AutoModelForCausalLM.from_pretrained(generator_model_name)

# Set padding token if needed
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
        print(f"✅ Loaded cached summary for '{movie_name}'")
    else:
        try:
            # Check for the exact Wikipedia page title
            if movie_name.lower() == "titanic":
                movie_name = "Titanic (1997 film)"
            text = wikipedia.page(movie_name).content
            print(f"🌐 Fetched summary for '{movie_name}' from Wikipedia")
        except wikipedia.DisambiguationError as e:
            print(f"Disambiguation error for {movie_name}. Try a more specific name. Options: {e.options}")
            return []
        except wikipedia.PageError:
            print("Movie not found on Wikipedia.")
            return []

        words = text.split()
        chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f)

    return chunks


def build_faiss_index(chunks):
    """Create FAISS index from text chunks."""
    embeddings = embedding_model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings, chunks


def retrieve_context(query, index, chunks, top_k=3):
    """Retrieve top-k most relevant chunks."""
    query_embedding = embedding_model.encode([query])
    _, indices = index.search(query_embedding, top_k)
    return " ".join([chunks[i] for i in indices[0]])


def generate_response(context, question):
    """Generate answer using a local language model."""
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
# 🚀 Main Program
# ---------------------------

def main():
    movie_name = input("🎬 Enter movie name: ").strip()
    question = input("❓ Ask your question about the plot: ")

    chunks = fetch_and_chunk(movie_name)
    if not chunks:
        return

    index, _, chunk_texts = build_faiss_index(chunks)
    context = retrieve_context(question, index, chunk_texts)
    answer = generate_response(context, question)

    print("\n💬 Answer:", answer)


if __name__ == "__main__":
    main()

