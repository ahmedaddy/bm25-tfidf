"""A small, dependency-free implementation of the Okapi BM25 ranking model."""

# Lets us use modern annotations such as ``list[str]`` on older Python versions.
from __future__ import annotations

from collections import Counter
from math import log
from typing import Callable, Iterable, Sequence


# A caller may supply raw text or a sequence of tokens it has already prepared.
Document = str | Sequence[str]
# Custom tokenizers take raw text and yield its tokens.
Tokenizer = Callable[[str], Iterable[str]]


def _tokens(value: Document, tokenizer: Tokenizer | None) -> list[str]:
    """Convert a document/query to tokens without mutating caller data."""
    if isinstance(value, str):
        # Default processing makes matching case-insensitive and splits on whitespace.
        return list(tokenizer(value) if tokenizer else value.lower().split())
    # Copy an input token sequence so the caller's object is never changed.
    return list(value)


class BM25:
    """Okapi BM25 index.

    Parameters follow the usual notation: ``k1`` controls term-frequency
    saturation and ``b`` controls document-length normalization.  Documents
    can be strings (whitespace-tokenized, lower-cased by default) or already
    tokenized sequences.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        if k1 < 0:
            raise ValueError("k1 must be non-negative")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")
        self.k1 = k1  # Higher values allow repeated terms to matter more.
        self.b = b  # 0 ignores document length; 1 fully normalizes for it.
        self.tokenizer = tokenizer
        self.documents: list[Document] = []  # Original documents, in result order.
        self.term_frequencies: list[Counter[str]] = []  # Word counts per document.
        self.document_lengths: list[int] = []  # Token count per document.
        self.idf: dict[str, float] = {}  # Rarity weight for every corpus term.
        self.average_document_length = 0.0  # Needed for BM25 length normalization.

    def fit(self, documents: Iterable[Document]) -> "BM25":
        """Build an index for *documents* and return this instance."""
        # Materialize iterators once: callers can pass a generator safely.
        self.documents = list(documents)
        tokenized = [_tokens(document, self.tokenizer) for document in self.documents]
        # Counter maps a term to how often it occurs in each document.
        self.term_frequencies = [Counter(tokens) for tokens in tokenized]
        self.document_lengths = [len(tokens) for tokens in tokenized]
        count = len(tokenized)
        # Empty corpora have an average length of zero rather than dividing by zero.
        self.average_document_length = (
            sum(self.document_lengths) / count if count else 0.0
        )

        # Document frequency counts in how many documents a term appears (not its
        # total number of occurrences); common terms should get lower IDF.
        document_frequency: Counter[str] = Counter()
        for frequencies in self.term_frequencies:
            document_frequency.update(frequencies.keys())
        # Robertson/Sparck Jones IDF. ``0.5`` smooths small collections and the
        # outer ``1 +`` keeps weights positive even for a corpus-wide term.
        self.idf = {
            term: log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }
        return self

    def score(self, query: Document) -> list[float]:
        """Return one BM25 score per fitted document, in input order."""
        if not self.documents:
            # There is no document to assign a score to before fitting a corpus.
            return []
        query_terms = _tokens(query, self.tokenizer)
        scores: list[float] = []
        # Avoid zero division when every fitted document is empty.
        average_length = self.average_document_length or 1.0
        for frequencies, length in zip(self.term_frequencies, self.document_lengths):
            # Longer documents need more occurrences to receive the same credit.
            normalization = self.k1 * (1.0 - self.b + self.b * length / average_length)
            score = 0.0
            # Repeated query terms deliberately contribute repeatedly, the
            # conventional BM25 query-frequency behavior.
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if frequency:
                    # BM25's core: IDF × saturated term frequency × length correction.
                    score += self.idf.get(term, 0.0) * frequency * (self.k1 + 1.0) / (
                        frequency + normalization
                    )
            scores.append(score)
        return scores

    # Alternate name used by several BM25 libraries.
    get_scores = score

    def search(self, query: Document, limit: int | None = None) -> list[tuple[int, float]]:
        """Return ``(document_index, score)`` pairs, highest score first."""
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative or None")
        # Sort by descending score; document index breaks ties predictably.
        ranked = sorted(enumerate(self.score(query)), key=lambda item: (-item[1], item[0]))
        return ranked if limit is None else ranked[:limit]


class BM25Okapi(BM25):
    """Compatibility name commonly used for the standard BM25 variant."""


def bm25_score(
    query: Document, documents: Iterable[Document], k1: float = 1.5, b: float = 0.75
) -> list[float]:
    """Convenience function for scoring a query against a one-off corpus."""
    return BM25(k1=k1, b=b).fit(documents).score(query)
