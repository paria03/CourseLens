# CourseLens — Dataset and Evaluation Plan

CourseLens is a study assistant that takes a student's question in plain language and retrieves the lecture material that answers it. This document describes the data we evaluate on, how we measure retrieval quality, and the initial results from our TF-IDF baseline.

## 1. Dataset Overview

The dataset covers two lectures from our NLP course:

- **Lecture 3:** `03 - Language Models.pdf` (language models, smoothing, perplexity)
- **Lecture 4:** `04 - Statistical Task Classification.pdf` (Naive Bayes, TF-IDF, logistic regression, evaluating classifiers)

It has two parts:

- **91 searchable chunks** (52 from Lecture 3, 39 from Lecture 4). Each chunk is the extracted text of one slide. Title and section-divider slides with no content of their own are not included (Lecture 3 slides 1, 2 and 26; Lecture 4 slides 1, 2 and 24).
- **30 evaluation questions**, each linked to the one chunk that answers it.

## 2. Dataset Structure

**`data/chunks.csv`**: the searchable lecture chunks.

| Column | Contents |
|---|---|
| `chunk_id` | Unique ID built from lecture and slide, e.g. `L03_p15` = Lecture 3, slide 15 |
| `text` | Text extracted from the slide |
| `document` | Source PDF file name |
| `page` | Slide number in the PDF |
| `n_words` | Number of words in `text` (8 to 126 per chunk, median 66) |

**`data/questions.csv`**: the evaluation questions.

| Column | Contents |
|---|---|
| `question_id` | `Q01` to `Q30` |
| `question` | The student-style question |
| `question_type` | `direct` or `paraphrased` (see section 3) |
| `document` | Lecture PDF that answers the question |
| `page` | Slide that answers the question |
| `correct_chunk_id` | The `chunk_id` from `chunks.csv` that answers the question |

The questions are balanced: 15 per lecture and 15 of each type. They are conceptual rather than calculation-based, and every one is answered by the lecture slides alone, with no outside knowledge needed.

## 3. Question Types

- **Direct** questions use the lecture's own terms. They check that the system can find material when the student already knows the right vocabulary.
- **Paraphrased** questions ask about the same ideas in everyday wording, the way a student who doesn't remember the term might ask. They test whether retrieval works on meaning rather than shared keywords.

Example pair, both answered by chunk `L03_p15` (the Markov assumption slide):

| | Question |
|---|---|
| Direct (Q04) | *What is the Markov assumption in n-gram language models?* |
| Paraphrased (Q05) | *Why do n-gram models only look at the previous one or two words instead of everything earlier in the sentence?* |

The dataset has four such pairs (Q01/Q02, Q04/Q05, Q17/Q18, Q25/Q26). Each pair shares the same `correct_chunk_id`, so the only difference between the two questions is the wording.

## 4. Evaluation Methodology

For each question, the retrieval system returns a ranked list of chunk IDs. A retrieved chunk counts as **correct** only if its `chunk_id` exactly matches the question's `correct_chunk_id`.

- **Top-1 accuracy**: the share of questions where the first result is the correct chunk.
  `Top-1 = (# questions where rank 1 = correct_chunk_id) / 30`
  *If the student reads only the first result, how often is it the right slide?*
- **Top-3 accuracy**: the share of questions where the correct chunk is anywhere in the first three results.
  `Top-3 = (# questions where correct_chunk_id is in ranks 1–3) / 30`
  *If the student scans a short list, how often is the answer in it?*

Top-3 is always at least as high as Top-1. A large gap between them means the system finds the right area but ranks a related slide first.

**Illustrative example** (made-up IDs, not a result):

| Question | correct_chunk_id | Retrieved (ranks 1, 2, 3) | Top-1 | Top-3 |
|---|---|---|---|---|
| Qa | C12 | C12, C40, C07 | hit | hit |
| Qb | C33 | C31, C33, C02 | miss | hit |
| Qc | C08 | C50, C51, C19 | miss | miss |

Here Top-1 = 1/3 and Top-3 = 2/3.

## 5. Evaluation Comparison

We report Top-1 and Top-3 for:

- **All 30 questions** (the headline numbers).
- **Direct vs paraphrased** (15 each). This is the main comparison. We expect a keyword-based retriever to do worse on paraphrased questions; whether a meaning-based retriever narrows that gap is a question for later experiments. The four pairs give a like-for-like check of the same answer under different wording.
- **Lecture 3 vs Lecture 4** (15 each), to spot a system that favours one lecture.

These are overlapping breakdowns of the same 30 questions, not separate datasets. For each miss we will also note what was retrieved instead: a related slide (e.g. a summary slide, or the same concept in the other lecture) or something unrelated.

## 6. Initial TF-IDF Results

Our first baseline represents each of the 91 chunks as a TF-IDF vector, ranks chunks by cosine similarity to the question, and scores the ranking against `correct_chunk_id` as described in section 4. Results on all 30 questions, as presented in Project Update 2:

| Question group | Questions | Top-1 | Top-3 |
|---|---|---|---|
| All questions | 30 | 8/30 (26.7%) | 16/30 (53.3%) |
| Direct | 15 | 6/15 (40.0%) | 12/15 (80.0%) |
| Paraphrased | 15 | 2/15 (13.3%) | 4/15 (26.7%) |
| Lecture 3 | 15 | 3/15 (20.0%) | 6/15 (40.0%) |
| Lecture 4 | 15 | 5/15 (33.3%) | 10/15 (66.7%) |

- **Direct questions did much better than paraphrased ones** (Top-3: 80.0% vs 26.7%). This fits what a keyword-weighting method would be expected to do: it finds the slide far more often when the question shares the slide's vocabulary.
- **Lecture 4 questions did better than Lecture 3 questions** in this run. With 15 questions per group, these gaps rest on a handful of questions and should be read as initial observations.
- **Next step:** the gap between direct and paraphrased questions motivates testing semantic retrieval, such as word embeddings, on the same 30 questions. These results do not show that such methods will perform better; that has to be measured.

## 7. Current Status

- **Dataset complete.** All 30 questions have a `correct_chunk_id`, and every one was verified by reading the chunk text, not just by matching document and page. A chunk was accepted only if it fully answers the question by itself. The supporting slide evidence for each question is recorded in a separate review document (`question_review.md`).
- **TF-IDF baseline and initial evaluation complete.** Top-1 and Top-3 results for all 30 questions are in section 6.
- **Still planned:**
  - a keyword-only baseline for comparison;
  - parameter tuning (no parameter search has been run yet);
  - error analysis of individual misses;
  - semantic retrieval experiments, such as word embeddings, on the same labeled questions.

## 8. Limitations

- **Small dataset.** With 30 questions, each question is worth about 3.3 percentage points overall and 6.7 points within a 15-question group, so small differences between systems or groups should not be over-interpreted.
- **Exact chunk-ID scoring and overlapping concepts.** Some ideas appear on more than one slide, such as Laplace smoothing and Bayes' Rule in both lectures, and the summary slides that restate earlier definitions. Because each question has a single correct chunk, retrieving a related slide counts as a miss even when it is relevant. We will flag these cases in the error analysis rather than count them as correct.
- **Text-extraction issues.** Chunks contain text extracted from the slides, so:
  - formulas that appear only as images are missing;
  - tables come out flattened;
  - on two-column slides, headings can end up separated from their text (e.g. the L1/L2 regularisation slide).

  We checked that every assigned chunk still answers its question, but these issues could affect how well a retrieval system matches some slides.
