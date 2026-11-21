import re
def normalize(text):
    return re.sub(r"\W+", " ", text.lower()).strip()
def is_exact_match(a, b):
    na, nb = normalize(a), normalize(b)
    return bool(na) and na == nb
