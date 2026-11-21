# SemantiCheck AI

English-only document comparison application using exact matching, TF-IDF and optional Sentence-BERT.

## Setup (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
python app.py
```

Open http://127.0.0.1:5000

Add your Gemini API key to `.env` as `GEMINI_API_KEY=...` to enable explanations.

## Dataset note
`train_snli.txt` is SNLI-style semantic inference data, not verified plagiarism labels. Do not use its labels as plagiarism ground truth. The PAN corpus should be used for plagiarism-specific training/evaluation after extraction and annotation parsing.

## Notes
- First Sentence-BERT use may download the model and require internet access.
- If Sentence-BERT cannot load, TF-IDF and exact matching still run.
- Scanned PDFs require OCR, which is not included.
- Compare documents against one another or a reference collection; this app currently implements two-document comparison.
- Similarity is evidence for review, not proof of plagiarism.


## Included NLP techniques
- Preprocessing: tokenization, lowercase conversion, stop-word removal, stemming and lemmatization (`models/preprocessing.py`).
- TF-IDF: lexical sentence-pair comparison.
- Word2Vec: can train a small local model from the compared documents or load a local Gensim model.
- GloVe: loads a pretrained Stanford GloVe text-vector file if installed in `embeddings/`.
- BERT: optional `bert-base-uncased` contextual encoder.
- Sentence-BERT: primary semantic encoder (`all-MiniLM-L6-v2`).
- Exact matching: normalized sentence comparison.

### Optional embedding setup
1. Word2Vec can be trained locally automatically when Gensim is installed and enough text is available. It saves to `embeddings/word2vec.model`.
2. For GloVe, download a vector archive from https://nlp.stanford.edu/projects/glove/, extract `glove.6B.100d.txt`, and place it at `embeddings/glove.6B.100d.txt`.
3. BERT and Sentence-BERT download pretrained weights on first use and need internet access. They are pretrained models, not trained from scratch by this application.
4. Optional models may be slow on CPU. The application continues with exact matching and TF-IDF if optional weights are unavailable.

## Dataset integrity
PAN corpus annotations should be the source of plagiarism-specific labels. SNLI is semantic inference data and must not be represented as ground-truth plagiarism training data.


## Easy setup inside the app
- Open `http://127.0.0.1:5000/settings`.
- Paste your Gemini API key and choose **Save API key**. It is stored locally in `.env`; the app never displays it back.
- Upload a dataset ZIP or a TXT/CSV/JSON/XML file. ZIP archives are safely extracted into `datasets/`.
- For PAN split RAR downloads: extract both parts with 7-Zip/WinRAR first, then ZIP the extracted corpus folder and upload it. The current app stores the dataset; PAN annotation parsing/training integration is a separate next step.
- SNLI can be uploaded as a data file for auxiliary semantic experiments, but its labels are not plagiarism labels.
- Dataset upload limit in this local starter is 512 MB. For larger corpora, copy the extracted dataset folder directly into `SemanticAi/datasets/`.

The UI uses a responsive premium-style white theme with indigo and violet accents. Google Fonts and Bootstrap are loaded from CDNs, so the external network is needed for those visual assets.
