Project: IR final project

Pre-run steps (run these before `streamlit run app.py`):

1. Create and activate a virtual environment
   - PowerShell:
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
   - CMD:
     python -m venv .venv
     .\.venv\Scripts\activate.bat

2. Install dependencies:
   pip install -r requirements.txt

3. Generate precomputed values (embeddings / eval caches) and build index:
   - Preferred (single step if script handles both):
     python precompute_model_evals.py
   - If the precompute script does not build the search index, run the project's index build step:
     - If there is a dedicated script:
       python build_index.py
     - Or run the index builder directly from package:
       python -c "from core.IR_core.search_pipeline import SearchPipeline; SearchPipeline().build_index()"
   Note: The built index (e.g., built_index/ or precomputed_index/) is required by the app and is large — it is ignored by git.

4. Run the app:
   streamlit run app.py

Notes:
- __pycache__ directories are created automatically by Python; you do not need to create them.
- Large precomputed files, model weights and data are excluded from version control by the provided .gitignore.

Brief description
- Data: Project expects a local data/ folder with documents, queries and relevance judgments (qrels). Precomputed embeddings, evaluation caches and built index files are stored under precomputed/ or built_index/ (ignored by git).
- Models used (high-level):
  - Sparse retrieval: TF-IDF, BM25
  - Dense retrieval: embedding models (e.g. sentence-transformers)
  - Re-ranker: cross-encoder style re-ranking
- Metrics used:
  - Mean Average Precision (MAP)
  - Mean Reciprocal Rank (MRR)
  - Recall@k, NDCG@k
- Features:
  - End-to-end search pipeline (indexing, retrieval, re-ranking)
  - Precomputation and caching of heavy model outputs
  - Evaluation utilities for comparing models