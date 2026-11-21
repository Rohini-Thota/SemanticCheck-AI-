
import os
import logging

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-3.8-flash"


def explain_matches(matches):

    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        return (
            "Gemini explanation unavailable because the API key "
            "is not configured. Local detection results are available."
        )

    if not matches:
        return (
            "No suspicious passages were detected by the local "
            "plagiarism detection engine. No individual passages "
            "were sent for Gemini analysis."
        )

    # Prepare evidence from detected passages
    evidence_parts = []

    for index, match in enumerate(matches[:5], start=1):

        evidence_parts.append(
            f"""
MATCH {index}

Document A - Reference:
{match.get('sentence_a', '')}

Document B - Compared document:
{match.get('sentence_b', '')}

Similarity score:
{match.get('score', 0)}%

Match category:
{match.get('category', 'Similarity')}

TF-IDF:
{match.get('tfidf', 'Not available')}%

Sentence-BERT:
{match.get('sbert', 'Not available')}%

Word2Vec:
{match.get('word2vec', 'Not available')}%

GloVe:
{match.get('glove', 'Not available')}%

BERT:
{match.get('bert', 'Not available')}%
"""
        )

    evidence = "\n".join(evidence_parts)

    prompt = f"""
You are an academic plagiarism and document similarity
analysis assistant for SemanticAi.

You are given matching passages detected by a local NLP system.
Your task is to explain the actual evidence clearly and accurately.

Analyze EACH match separately.

For every match, use this exact structure:

MATCH NUMBER:

1. MATCH TYPE
Identify whether the passage appears to contain:
- Exact copying
- Partial copying
- Possible paraphrasing
- Semantic similarity

2. DOCUMENT A - IMPORTANT CONTENT
Identify the important words, phrases, and ideas in the
reference passage.

3. DOCUMENT B - IMPORTANT CONTENT
Identify the important words, phrases, and ideas in the
compared passage.

4. WHAT IS SIMILAR?
Explain which words, phrases, concepts, or meanings overlap.

5. WHAT HAS CHANGED?
Identify wording that has been changed, shortened,
rearranged, or paraphrased, if applicable.

6. WHAT IS THE POSSIBLE ISSUE?
Explain why the passage was flagged and whether it appears
to be copied, partially copied, or semantically similar.

7. RECOMMENDATION
Explain what the student should check, including original
sources, citations, quotations, and attribution.

After analyzing all matches, provide:

OVERALL ASSESSMENT:
Summarize the main findings across the detected passages.

IMPORTANT RULES:

- Analyze only the evidence provided below.
- Never invent missing text or claim to have accessed an
  external source.
- Do not call ordinary shared technical terms plagiarism
  without additional evidence.
- Do not assume that similar meaning automatically means copying.
- Do not declare a student guilty of plagiarism.
- Explain uncertainty where the evidence is insufficient.
- Distinguish exact wording overlap from paraphrased meaning.
- Use clear, professional academic language.
- Make the explanation detailed but easy for a student to understand.

DETECTED EVIDENCE:

{evidence}
"""

    client = None

    try:

        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=20000)
        )

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=1200
            )
        )

        explanation = getattr(response, "text", None)

        if explanation and explanation.strip():
            return explanation.strip()

        logger.warning("Gemini returned an empty response.")

        return (
            "Gemini returned an empty explanation. "
            "Please review the detected passages manually."
        )

    except Exception as error:

        # Print complete error details in the terminal
        logger.exception("Gemini API error occurred")

        # Display actual error on the results page
        return (
            f"Gemini explanation failed: "
            f"{type(error).__name__}: {str(error)}. "
            "Your local plagiarism detection results are still available."
        )

    finally:

        if client is not None:
            try:
                client.close()
            except Exception:
                logger.warning("Could not close Gemini client.")
