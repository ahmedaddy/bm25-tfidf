"""A small, dependency-free TF-IDF vectorizer and cosine-search index."""

# Lets us use modern annotations such as ``list[str]`` on older Python versions.
from __future__ import annotations

from collections import Counter
from math import log, sqrt
from typing import Callable, Iterable, Sequence


# Inputs can be raw text or pre-tokenized word sequences.
Document = str | Sequence[str]
Tokenizer = Callable[[str], Iterable[str]]


def _tokens(value: Document, tokenizer: Tokenizer | None) -> list[str]:
    """Turn an input into tokens, using simple lower-cased splitting by default."""
    if isinstance(value, str):
        return list(tokenizer(value) if tokenizer else value.lower().split())
    # Return a copy so vectorization does not alter a caller's token sequence.
    return list(value)


class TFIDF:
    """TF-IDF vectorizer using smoothed IDF and L2-normalized vectors.

    Term frequency is ``count / document_length``.  IDF is
    ``log((1 + N) / (1 + df)) + 1``, so terms found in all documents retain a
    useful weight of one.  ``score`` returns cosine similarities.
    """

    def __init__(self, tokenizer: Tokenizer | None = None) -> None:
        self.tokenizer = tokenizer
        self.documents: list[Document] = []  # Original corpus, in result order.
        self.vocabulary: dict[str, int] = {}  # Stable term-to-column mapping.
        self.idf: dict[str, float] = {}  # Learned inverse document frequencies.
        self.document_vectors: list[dict[str, float]] = []  # Sparse normalized vectors.

    def fit(self, documents: Iterable[Document]) -> "TFIDF":
        """Learn vocabulary/IDF and store normalized document vectors."""
        # Store a generator's values so they can be used for fitting and search.
        self.documents = list(documents)
        tokenized = [_tokens(document, self.tokenizer) for document in self.documents]
        # Count words in each document to calculate its term frequencies.
        frequencies = [Counter(tokens) for tokens in tokenized]
        # Count the number of documents containing each word, once per document.
        document_frequency: Counter[str] = Counter()
        for frequency in frequencies:
            document_frequency.update(frequency.keys())

        total = len(tokenized)  # N: number of documents in the corpus.
        # Sorting makes column indices deterministic across runs.
        self.vocabulary = {term: index for index, term in enumerate(sorted(document_frequency))}
        # Add-one smoothing prevents division by zero and gives every known term
        # a positive weight, including terms that appear in every document.
        self.idf = {
            term: log((1.0 + total) / (1.0 + frequency)) + 1.0
            for term, frequency in document_frequency.items()
        }
        # Convert every document to the representation used during cosine search.
        self.document_vectors = [self._vectorize_counts(counts) for counts in frequencies]
        return self

    def _vectorize_counts(self, counts: Counter[str]) -> dict[str, float]:
        # Term frequency is occurrence count divided by document length.
        length = sum(counts.values())
        if not length:
            # Empty documents have no direction and therefore an empty vector.
            return {}
        vector = {
            # Ignore words unseen in fit(): they have no learned IDF weight.
            term: count / length * self.idf[term]
            for term, count in counts.items()
            if term in self.idf
        }
        # L2 normalization makes a dot product equal cosine similarity.
        norm = sqrt(sum(weight * weight for weight in vector.values()))
        return {term: weight / norm for term, weight in vector.items()} if norm else {}

    def transform(self, documents: Iterable[Document]) -> list[dict[str, float]]:
        """Convert documents to normalized sparse TF-IDF vectors."""
        # Counter first converts tokens to term counts; the helper applies TF-IDF.
        return [self._vectorize_counts(Counter(_tokens(document, self.tokenizer))) for document in documents]

    def fit_transform(self, documents: Iterable[Document]) -> list[dict[str, float]]:
        """Fit the index and return its document vectors."""
        self.fit(documents)
        return self.document_vectors

    def score(self, query: Document) -> list[float]:
        """Return cosine similarities between *query* and each document."""
        # Queries use the same vocabulary and IDF learned from the corpus.
        query_vector = self._vectorize_counts(Counter(_tokens(query, self.tokenizer)))
        return [
            # Dot product of two L2-normalized sparse vectors = cosine similarity.
            sum(query_vector.get(term, 0.0) * weight for term, weight in vector.items())
            for vector in self.document_vectors
        ]

    # Alternate name used by ranking-oriented callers.
    get_scores = score

    def search(self, query: Document, limit: int | None = None) -> list[tuple[int, float]]:
        """Return ``(document_index, similarity)`` pairs, highest first."""
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative or None")
        # Higher similarity comes first; index resolves equal scores predictably.
        ranked = sorted(enumerate(self.score(query)), key=lambda item: (-item[1], item[0]))
        return ranked if limit is None else ranked[:limit]


class TfidfVectorizer(TFIDF):
    """Compatibility alias for callers expecting a vectorizer-style name."""


def tf_idf(query: Document, documents: Iterable[Document]) -> list[float]:
    """Convenience function returning query-to-document TF-IDF similarities."""
    return TFIDF().fit(documents).score(query)
