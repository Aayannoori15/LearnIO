# Project Structure

This project now uses one shared backend environment at the project root.

- `app.py`, `home.html`, `styles.css`, and `saved_summary_model/` power the FastAPI app.
- `rag/` contains the RAG notebook, pipeline code, and data.
- `text_summarizer/` contains the training notebook.
- `requirements.txt` is the single backend dependency list for both summarization and RAG.

Setup the shared environment:

```bash
cd "/Users/aayannoori/Desktop/Deep Learning /TextSummarizer"
/opt/anaconda3/envs/smartgate/bin/python -m venv --copies .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the app:

```bash
cd "/Users/aayannoori/Desktop/Deep Learning /TextSummarizer"
source .venv/bin/activate
uvicorn app:app --reload
```

Open notebooks from the same shared environment:

```bash
cd "/Users/aayannoori/Desktop/Deep Learning /TextSummarizer"
source .venv/bin/activate
jupyter notebook
```

Recommended kernel:

- `Python (backend-env)` for both `rag/RAG.ipynb` and `text_summarizer/textsummarizer.ipynb`
