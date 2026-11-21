from models.preprocessing import split_sentences, preprocess

def sentence_pairs_from_documents(source_text, suspicious_text):
    source = split_sentences(source_text)
    suspicious = split_sentences(suspicious_text)
    return [(s, t) for s in suspicious for t in source]
