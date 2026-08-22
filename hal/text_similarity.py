"""Literal (surface) similarity -- purely algorithmic, no LLM involved.

This mirrors ``jacc()`` in the original ``evaluation.py``: NLTK word
tokenization, English stop-word removal, lower-casing, then Jaccard similarity
between the two token *sets*.

NLTK is used when it is installed and its corpora are available (that is the
faithful path).  Otherwise a regex tokenizer with NLTK's English stop-word list
is used so that the metric still runs in a bare environment; the fallback is
reported by :func:`tokenizer_backend`.
"""

from __future__ import annotations

from typing import List, Set

# NLTK's English stop-word list (used verbatim by the fallback tokenizer).
_FALLBACK_STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "you're",
    "you've", "you'll", "you'd", "your", "yours", "yourself", "yourselves", "he",
    "him", "his", "himself", "she", "she's", "her", "hers", "herself", "it",
    "it's", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "that'll", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and",
    "but", "if", "or", "because", "as", "until", "while", "of", "at", "by",
    "for", "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "to", "from", "up", "down", "in",
    "out", "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "s", "t", "can",
    "will", "just", "don", "don't", "should", "should've", "now", "d", "ll",
    "m", "o", "re", "ve", "y", "ain", "aren", "aren't", "couldn", "couldn't",
    "didn", "didn't", "doesn", "doesn't", "hadn", "hadn't", "hasn", "hasn't",
    "haven", "haven't", "isn", "isn't", "ma", "mightn", "mightn't", "mustn",
    "mustn't", "needn", "needn't", "shan", "shan't", "shouldn", "shouldn't",
    "wasn", "wasn't", "weren", "weren't", "won", "won't", "wouldn", "wouldn't",
}

_backend = None
_stopwords: Set[str] = set()
_word_tokenize = None


def _try_nltk() -> bool:
    """Wire up the NLTK tokenizer; ``False`` if it is not usable."""
    from nltk.corpus import stopwords as nltk_stopwords
    from nltk.tokenize import word_tokenize as nltk_word_tokenize

    global _stopwords, _word_tokenize, _backend
    words = set(nltk_stopwords.words("english"))
    nltk_word_tokenize("probe sentence.")  # triggers the punkt lookup
    _stopwords = words
    _word_tokenize = nltk_word_tokenize
    _backend = "nltk"
    return True


def _download_nltk_data() -> None:
    """Fetch the corpora NLTK needs, quietly and only once.

    The metric's literal-similarity term depends on which tokenizer runs, so a
    machine that silently falls back to the regex tokenizer produces slightly
    different Jaccard values -- and therefore slightly different MDS -- than one
    with NLTK data present. Downloading automatically removes a manual setup
    step that is easy to forget when moving the project to a new machine.
    """
    import nltk

    for package in ("punkt", "punkt_tab", "stopwords"):
        try:
            nltk.download(package, quiet=True, raise_on_error=False)
        except Exception:
            pass


def _init() -> None:
    global _backend, _stopwords, _word_tokenize
    if _backend is not None:
        return
    try:  # pragma: no cover - depends on the environment
        return None if _try_nltk() else None
    except Exception:
        pass
    try:  # pragma: no cover - one automatic attempt to fetch the missing data
        _download_nltk_data()
        return None if _try_nltk() else None
    except Exception:
        pass
    import re

    token_re = re.compile(r"\w+|[^\w\s]")
    _stopwords = set(_FALLBACK_STOPWORDS)
    _word_tokenize = token_re.findall
    _backend = "regex"


def tokenizer_backend() -> str:
    """``"nltk"`` (faithful to the original) or ``"regex"`` (fallback)."""
    _init()
    return _backend or "regex"


def tokenize(text: str) -> List[str]:
    _init()
    return list(_word_tokenize(text or ""))


def content_tokens(text: str) -> Set[str]:
    _init()
    return {w.lower() for w in tokenize(text) if w.lower() not in _stopwords}


def jaccard(text1: str, text2: str) -> float:
    """Jaccard similarity after tokenization and stop-word removal.

    Identical in structure to ``jacc()`` in the original ``evaluation.py``.
    """
    set1 = content_tokens(text1)
    set2 = content_tokens(text2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    if union == 0:
        return 0.0
    return intersection / union


# The original evaluation.py imports this name; keep it available as an alias.
jacc = jaccard
