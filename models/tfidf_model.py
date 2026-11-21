from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def pairwise_similarity(sentences_a, sentences_b):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform(list(sentences_a) + list(sentences_b))
    return cosine_similarity(matrix[:len(sentences_a)], matrix[len(sentences_a):])
