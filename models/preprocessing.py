import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

def ensure_nltk():
    import nltk
    for package in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
        try:
            nltk.download(package, quiet=True)
        except Exception:
            pass

def split_sentences(text):
    ensure_nltk()
    try:
        return [s.strip() for s in sent_tokenize(text) if s.strip()]
    except Exception:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]

def preprocess(text, remove_stopwords=True, stem=False, lemmatize=True):
    ensure_nltk()
    tokens = [t.lower() for t in word_tokenize(text) if re.search(r"[a-zA-Z]", t)]
    if remove_stopwords:
        try:
            stops = set(stopwords.words("english"))
            tokens = [t for t in tokens if t not in stops]
        except LookupError:
            pass
    if lemmatize:
        lemma = WordNetLemmatizer()
        try:
            tokens = [lemma.lemmatize(t) for t in tokens]
        except LookupError:
            pass
    if stem:
        ps = PorterStemmer()
        tokens = [ps.stem(t) for t in tokens]
    return tokens
