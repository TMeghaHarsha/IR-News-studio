# News IR Studio

Two short sections below explain how to run the app and what the project contains.

## Section 1 — How to use the app

1) Download the files as ZIP

2) Extract them to a folder

3) Open the folder in your file explorer, right-click on an empty area and choose "Open in Terminal" (this launches PowerShell in the project folder)

4) Create and activate a virtual environment (recommended):

   - Create: `python -m venv .venv`
   - Activate: `.venv\Scripts\Activate`

5) Upgrade pip and install dependencies:

   - `python -m pip install --upgrade pip`
   - `pip install -r requirements.txt` (this might take few minutes)

6) Download required NLTK data:

   - `python scripts\download_nltk.py`

7) Make sure index artifacts are available. Two options:

     - (Clone if you extracted ZIP):  
       `git clone https://github.com/TMeghaHarsha/IR-News-studio.git`  (this might take a few minutes)
       `cd IR-News-studio`

     - Initialize and fetch LFS objects:  
       `git lfs install`  
       `git lfs fetch --all`  
       `git lfs checkout`  
       (or: `git lfs pull origin main`)

     - Verify the files (PowerShell):  
       `Get-Content core\results\built_index\config_used.json -TotalCount 5`  (should print JSON beginning with `{`)  

8) (Optional) Run baseline evaluations (TF-IDF + BM25):

   `python -m core.IR_evaluation.run_experiments --indexdir core/results/built_index --queries data/queries_relevance_auto.json --output core/results/evaluation_results`

9) Quick smoke test (console):

   `python test_rankers.py`

10) Start the Streamlit UI (recommended):

   `streamlit run app.py -- --index-base core/results --qrels data/queries_relevance_auto.json`


## Section 2 — Project details

- Purpose: an end-to-end information retrieval studio for a news dataset (indexing, ranking, evaluation, and a Streamlit UI).
- Core features:
   - Inverted index builder and preprocessing pipeline.
   - Rankers: BM25, TF-IDF, query-likelihood language model, temporal BM25, and a Learning-to-Rank (XGBoost) pipeline.
   - Boolean query engine (AND/OR/NOT, quoted literals, parentheses).
   - Streamlit UI ("News IR Studio") with tabs for Ranked Search, Boolean Lab, Dataset overview, Live Evaluation, and model evaluation.
   - RAG answer generation using a HuggingFace seq2seq model (default: `google/flan-t5-base`).
   - Experiment runner and evaluation metrics (Precision, Recall, nDCG, MRR) and utilities to train LTR models.

- Quick pointers:
   - `core/build_index/build_index.py` builds index artifacts under `core/results/built_index`.
   - `core/IR_evaluation/run_experiments.py` runs the automatic TF-IDF/BM25 experiments.
   - `test_rankers.py` is a small console smoke test.
