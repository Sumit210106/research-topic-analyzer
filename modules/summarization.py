import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
import string


try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")


def extractive_summary(text, num_sentences=2):
    """
    Frequency-based extractive summarization
    """

    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""

    sentences = sent_tokenize(text)

    words = word_tokenize(text.lower())

    words = [w for w in words if w not in string.punctuation]

    freq = Counter(words)

    max_freq = max(freq.values())
    freq = {w: f / max_freq for w, f in freq.items()}

    sent_scores = {}
    for sent in sentences:
        sent_words = word_tokenize(sent.lower())
        score = sum(freq.get(w, 0) for w in sent_words)
        sent_scores[sent] = score

    summary_sentences = sorted(
        sent_scores,
        key=sent_scores.get,
        reverse=True
    )[:num_sentences]

    return " ".join(summary_sentences)