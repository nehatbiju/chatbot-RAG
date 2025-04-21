# 🎬 Movie Plot QA Bot

An interactive Streamlit app that uses Wikipedia + RAG (Retrieval-Augmented Generation) to answer questions about movie plots.

Ask things like:
- *What happened to Jack in Titanic?*
- *Who is the villain in Inception?*

## How It Works

- **Wikipedia**: Fetches movie summaries
- **Sentence Transformers**: Converts text to embeddings
- **FAISS**: Finds the most relevant chunks
- **GPT-2**: Generates answers based on context

##  Features

-  Wikipedia-powered knowledge
-  Semantic search via FAISS
-  GPT-2-based answer generation
- 🖥 Clean Streamlit UI

##  Installation

```bash
git clone https://github.com/yourusername/movie-plot-qa-bot.git
cd movie-plot-qa-bot
pip install -r requirements.txt

