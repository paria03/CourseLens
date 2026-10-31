# CourseLens
KAP: Krishna, Angelina, Paria

## Run the TF-IDF baseline

Use Python 3.11 or newer. From the repository folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python tfidf_retrieval.py "What is the Markov assumption?"
python evaluate_tfidf.py
```

`tfidf_retrieval.py` reads `data/chunks.csv`, fits a TF-IDF vectorizer on the
passage text, transforms your question with that same vectorizer, and ranks
passages by cosine similarity. Results include the source document and page.
Use `--top-k 5` to request five passages. This script retrieves existing text;
it does not generate an answer.

To follow the code, start with these functions in `tfidf_retrieval.py`:

1. `load_chunks()` reads the lecture CSV.
2. `build_index(chunks)` converts the lecture text into TF-IDF vectors.
3. `search(...)` converts a question into a vector, compares it with each
   passage, and returns the highest scores.

`main()` runs these steps and prints the matches. Once that makes sense, read
the question loop in `evaluate_tfidf.py` to see how each match is checked
against its labeled answer.

The initial baseline uses lowercase word unigrams, including one-character
tokens, smoothed IDF, raw term counts and L2 normalization. It keeps stopwords
and does not stem words. Scikit-learn's smoothed IDF is
`ln((1 + number_of_passages) / (1 + document_frequency)) + 1`.
Only passage text contributes to ranking. IDs, source pages and `n_words` are
metadata. Similarity scores are not probabilities of correctness. Queries with
no vocabulary overlap return no matches, and zero-score passages are omitted.

`evaluate_tfidf.py` applies the same baseline to all labeled questions. It saves
Top-1 and Top-3 metrics in `results/tfidf_metrics.json` and per-question rankings
in `results/tfidf_predictions.csv`, including breakdowns by question type and
source lecture. A hit requires the exact labeled chunk ID, as specified in
`docs/evaluation_plan.md`.

Top-1 checks whether the first result is correct. Top-3 checks whether any of
the first three is correct. Precision@3 instead divides the number of relevant
results by three; with a single labeled relevant chunk it is at most 1/3.
We report Top-1 and Top-3 for this baseline.

These are initial results on a small evaluation set, not evidence of general
performance. Keep a separate held-out set if tuning preprocessing or ranking
against these questions. TF-IDF does not guarantee semantic understanding or
that a passage supports an answer. Inspect incorrect results before changing
the method.

Implementation reference: [scikit-learn TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).
