# RAG

RAG now uses the shared root backend environment.

Setup:

```bash
cd "/Users/aayannoori/Desktop/Deep Learning /TextSummarizer"
source .venv/bin/activate
jupyter notebook
```

Notes:

- Open `rag/RAG.ipynb` with the `Python (backend-env)` kernel.
- Put your Groq API key in `rag/.env` based on `.env.example`.
- The notebook expects text files inside `rag/data/`.
