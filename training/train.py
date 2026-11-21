"""Train a baseline sentence-pair classifier only when a correctly labelled CSV is supplied.
Expected columns: sentence1, sentence2, label. SNLI labels are semantic inference labels,
not plagiarism labels; do not describe this model as a plagiarism classifier.
"""
import argparse, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--text1", default="sentence1")
    parser.add_argument("--text2", default="sentence2")
    parser.add_argument("--label", default="label")
    parser.add_argument("--output", default="trained_models/pair_classifier.joblib")
    args = parser.parse_args()
    df = pd.read_csv(args.csv)
    df = df.dropna(subset=[args.text1, args.text2, args.label])
    X = (df[args.text1].astype(str) + " [SEP] " + df[args.text2].astype(str))
    y = df[args.label]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    model = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=150000)),
                      ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    model.fit(Xtr, ytr)
    print(classification_report(yte, model.predict(Xte)))
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    joblib.dump(model, args.output)
    print("Saved:", args.output)
if __name__ == "__main__":
    main()
