"""Search lecture slides using TF-IDF and cosine similarity."""

import argparse
import csv
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_FOLDER = Path(__file__).resolve().parent


def load_chunks():
    file_path = PROJECT_FOLDER / "data" / "chunks.csv"
    with file_path.open(encoding="utf-8-sig", newline="") as file:
        chunks = list(csv.DictReader(file))

    if not chunks:
        raise ValueError("There are no lecture chunks in chunks.csv.")

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Each chunk needs a unique chunk_id.")
    return chunks


def build_index(chunks):
    texts = []
    for chunk in chunks:
        texts.append(chunk["text"])

    # Use single words, keep stopwords, and include one-letter tokens like k.
    # The other settings use sklearn's defaults: smoothed IDF and L2 normalization.
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    chunk_vectors = vectorizer.fit_transform(texts)
    return vectorizer, chunk_vectors


def search(question, chunks, vectorizer, chunk_vectors, top_k=3):
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if not question.strip():
        return []

    # Reuse the lecture vocabulary and weights instead of fitting on the question.
    question_vector = vectorizer.transform([question])
    scores = cosine_similarity(question_vector, chunk_vectors)[0]

    matches = []
    for chunk, score in zip(chunks, scores):
        if score > 0:
            match = chunk.copy()
            match["score"] = float(score)
            matches.append(match)

    # Sort by ID first so equal scores always have the same order.
    matches.sort(key=lambda match: match["chunk_id"])
    matches.sort(key=lambda match: match["score"], reverse=True)
    return matches[:top_k]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Put your question in quotes.")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    chunks = load_chunks()
    vectorizer, chunk_vectors = build_index(chunks)
    matches = search(args.question, chunks, vectorizer, chunk_vectors, args.top_k)

    print(f"Loaded {len(chunks)} lecture passages.")
    if not matches:
        print("No vocabulary overlap found. Try rewording your question.")

    for rank, match in enumerate(matches, start=1):
        print(f"\n{rank}. {match['chunk_id']} (similarity: {match['score']:.4f})")
        print(f"Source: {match['document']}, page {match['page']}")
        print(match["text"])


if __name__ == "__main__":
    main()
