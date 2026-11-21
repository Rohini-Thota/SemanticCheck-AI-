
import os
import json
from exact_match import normalize
from tfidf_model import pairwise_similarity
from preprocessing import split_sentences


APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS_PATH = os.path.join(APP_DIR, "plagiarism_settings.json")


DEFAULT_SETTINGS = {
    "similarity_threshold": 75,
    "detect_exact": True,
    "detect_partial": True,
    "detect_paraphrase": True,
    "detect_semantic": True,
    "comparison_level": "document",
    "highlight_matches": True,
    "report_matches": True,
    "report_scores": True,
    "report_recommendations": True,
    "delete_after_analysis": False
}


def load_settings():
    settings = DEFAULT_SETTINGS.copy()

    try:
        if os.path.exists(SETTINGS_PATH):
            with open(SETTINGS_PATH, "r", encoding="utf-8") as file:
                saved = json.load(file)

            if isinstance(saved, dict):
                settings.update(saved)

    except (OSError, json.JSONDecodeError):
        pass

    try:
        settings["similarity_threshold"] = max(
            10, min(100, int(settings["similarity_threshold"]))
        )
    except (TypeError, ValueError):
        settings["similarity_threshold"] = 75

    return settings


def safe_score(matrix, i, j):
    try:
        value = float(matrix[i][j])

        if value != value:
            return 0.0

        return max(0.0, min(1.0, value))

    except (TypeError, ValueError, IndexError):
        return 0.0


def compare_documents(text_a, text_b):

    settings = load_settings()

    a = split_sentences(text_a)
    b = split_sentences(text_b)

    if not a or not b:
        raise ValueError(
            "One of the documents has no extractable sentences."
        )

    # --------------------------------------------------
    # TF-IDF
    # --------------------------------------------------

    sim = pairwise_similarity(a, b)

    # --------------------------------------------------
    # OPTIONAL SEMANTIC MODELS
    # --------------------------------------------------

    sbert = None

    optional = {
        "word2vec": None,
        "glove": None,
        "bert": None
    }

    # --------------------------------------------------
    # SENTENCE-BERT
    # --------------------------------------------------

    try:
        from models.sbert_model import SBERTEncoder

        encoder = SBERTEncoder()

        embeddings_a = encoder.encode(a)
        embeddings_b = encoder.encode(b)

        sbert = embeddings_a @ embeddings_b.T

    except Exception:
        sbert = None

    # --------------------------------------------------
    # WORD2VEC
    # Changed: Do not retrain Word2Vec during comparison.
    # Load existing model only; skip if unavailable.
    # --------------------------------------------------

    try:
        from models.word2vec_model import Word2VecEncoder

        encoder = Word2VecEncoder()

        try:
            encoder.load()

            optional["word2vec"] = [
                [encoder.similarity(x, y) for y in b]
                for x in a
            ]

        except Exception:
            optional["word2vec"] = None

    except Exception:
        optional["word2vec"] = None

    # --------------------------------------------------
    # GLOVE
    # --------------------------------------------------

    try:
        from models.glove_model import GloVeEncoder

        encoder = GloVeEncoder()
        encoder.load()

        optional["glove"] = [
            [encoder.similarity(x, y) for y in b]
            for x in a
        ]

    except Exception:
        optional["glove"] = None

    # --------------------------------------------------
    # BERT
    # --------------------------------------------------

    try:
        from models.bert_model import BERTEncoder

        encoder = BERTEncoder()

        embeddings_a = encoder.encode(a)
        embeddings_b = encoder.encode(b)

        optional["bert"] = embeddings_a @ embeddings_b.T

    except Exception:
        optional["bert"] = None

    # --------------------------------------------------
    # MATCH DETECTION
    # --------------------------------------------------

    matches = []
    scores = []

    b_norm = {
        normalize(sentence): index
        for index, sentence in enumerate(b)
    }

    threshold = settings["similarity_threshold"] / 100.0

    for i, sentence in enumerate(a):

        # Find the best matching sentence in document B
        j = max(
            range(len(b)),
            key=lambda k: safe_score(sim, i, k)
        )

        tf_score = safe_score(sim, i, j)

        sbert_score = (
            safe_score(sbert, i, j)
            if sbert is not None else None
        )

        word2vec_score = (
            safe_score(optional["word2vec"], i, j)
            if optional["word2vec"] is not None else None
        )

        glove_score = (
            safe_score(optional["glove"], i, j)
            if optional["glove"] is not None else None
        )

        bert_score = (
            safe_score(optional["bert"], i, j)
            if optional["bert"] is not None else None
        )

        normalized_sentence = normalize(sentence)

        exact = (
            len(normalized_sentence) > 20
            and normalized_sentence in b_norm
        )

        # --------------------------------------------------
        # COMBINE AVAILABLE MODEL SCORES
        # --------------------------------------------------

        semantic_scores = []

        if settings["detect_semantic"]:

            if sbert_score is not None:
                semantic_scores.append(sbert_score)

            if word2vec_score is not None:
                semantic_scores.append(word2vec_score)

            if glove_score is not None:
                semantic_scores.append(glove_score)

            if bert_score is not None:
                semantic_scores.append(bert_score)

        semantic_score = (
            sum(semantic_scores) / len(semantic_scores)
            if semantic_scores else tf_score
        )

        if settings["detect_semantic"] and semantic_scores:
            combined = (0.35 * tf_score) + (0.65 * semantic_score)
        else:
            combined = tf_score

        if exact:
            combined = 1.0

        combined = max(0.0, min(1.0, combined))

        # --------------------------------------------------
        # CLASSIFY MATCH
        # --------------------------------------------------

        category = "Low similarity"
        level = "low"

        if exact and settings["detect_exact"]:
            category = "Exact match"
            level = "high"

        elif combined >= threshold:

            if (
                settings["detect_paraphrase"]
                and semantic_scores
            ):
                category = "Strong possible paraphrase"
                level = "high"

            elif settings["detect_partial"]:
                category = "Possible partial similarity"
                level = "moderate"

        elif (
            settings["detect_partial"]
            and combined >= max(0.30, threshold - 0.20)
        ):
            category = "Possible partial similarity"
            level = "moderate"

        scores.append(combined)

        # Add suspicious matches
        if level != "low":

            matches.append({
                "sentence_a": sentence,
                "sentence_b": b[j],

                "sentence_number": i + 1,
                "matched_sentence_number": j + 1,

                "exact": exact,

                "tfidf": round(tf_score * 100, 2),

                "sbert": (
                    round(sbert_score * 100, 2)
                    if sbert_score is not None else None
                ),

                "word2vec": (
                    round(word2vec_score * 100, 2)
                    if word2vec_score is not None else None
                ),

                "glove": (
                    round(glove_score * 100, 2)
                    if glove_score is not None else None
                ),

                "bert": (
                    round(bert_score * 100, 2)
                    if bert_score is not None else None
                ),

                "score": round(combined * 100, 2),
                "category": category,
                "level": level
            })

    # --------------------------------------------------
    # OVERALL SIMILARITY
    # --------------------------------------------------

    overall = (
        round(sum(scores) / len(scores) * 100, 2)
        if scores else 0.0
    )

    exact_count = sum(
        1 for match in matches if match["exact"]
    )

    high_count = sum(
        1 for match in matches if match["level"] == "high"
    )

    suspicious_count = len(matches)

    # --------------------------------------------------
    # RISK ASSESSMENT
    # --------------------------------------------------

    if exact_count > 0 or high_count > 0:
        risk = "Potential plagiarism detected"

    elif suspicious_count > 0:
        risk = "Manual review recommended"

    else:
        risk = "No significant matching passages found"

    # --------------------------------------------------
    # MODEL STATUS
    # --------------------------------------------------

    model_status = {
        "TF-IDF": "active",

        "Sentence-BERT": (
            "active" if sbert is not None else "unavailable"
        ),

        "Word2Vec": (
            "active"
            if optional["word2vec"] is not None
            else "pretrained model unavailable"
        ),

        "GloVe": (
            "active"
            if optional["glove"] is not None
            else "pretrained file not installed"
        ),

        "BERT": (
            "active"
            if optional["bert"] is not None
            else "model unavailable"
        )
    }

    # --------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------

    return {
        "overall_similarity": overall,
        "risk": risk,

        "exact_count": exact_count,
        "suspicious_count": suspicious_count,
        "total_sentences": len(a),

        "matches": matches,

        "sbert_available": sbert is not None,

        "model_status": model_status,

        "settings_used": {
            "similarity_threshold": settings["similarity_threshold"],
            "comparison_level": settings["comparison_level"],
            "detect_exact": settings["detect_exact"],
            "detect_partial": settings["detect_partial"],
            "detect_paraphrase": settings["detect_paraphrase"],
            "detect_semantic": settings["detect_semantic"]
        },

        "method_note": (
            "Similarity scores indicate textual overlap or semantic "
            "similarity. They are evidence for review, not conclusive "
            "proof of plagiarism."
        )
    }
