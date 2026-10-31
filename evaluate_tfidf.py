"""Check how often TF-IDF retrieves the labeled slide for each question."""

import csv
import json

import sklearn

from tfidf_retrieval import PROJECT_FOLDER, build_index, load_chunks, search


def calculate_accuracy(predictions):
    top1_count = 0
    top3_count = 0
    for prediction in predictions:
        top1_count += prediction["top1_hit"]
        top3_count += prediction["top3_hit"]

    total = len(predictions)
    return {
        "n_questions": total,
        "top1_hits": top1_count,
        "top3_hits": top3_count,
        "top1_accuracy": top1_count / total,
        "top3_accuracy": top3_count / total,
    }


def main():
    chunks = load_chunks()
    vectorizer, chunk_vectors = build_index(chunks)

    question_file = PROJECT_FOLDER / "data" / "questions.csv"
    with question_file.open(encoding="utf-8-sig", newline="") as file:
        questions = list(csv.DictReader(file))
    if not questions:
        raise ValueError("There are no questions in questions.csv.")

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    predictions = []
    for question in questions:
        correct_id = question["correct_chunk_id"]
        if correct_id not in chunk_ids:
            raise ValueError(f"The labeled chunk {correct_id} does not exist.")

        matches = search(question["question"], chunks, vectorizer, chunk_vectors)
        retrieved_ids = []
        scores = []
        for match in matches:
            retrieved_ids.append(match["chunk_id"])
            scores.append(f"{match['score']:.6f}")

        # Top-1 checks the first slide. Top-3 checks all three returned slides.
        top1_hit = 0
        if retrieved_ids and retrieved_ids[0] == correct_id:
            top1_hit = 1
        top3_hit = int(correct_id in retrieved_ids)

        prediction = question.copy()
        prediction["retrieved_chunk_ids"] = ";".join(retrieved_ids)
        prediction["similarity_scores"] = ";".join(scores)
        prediction["top1_hit"] = top1_hit
        prediction["top3_hit"] = top3_hit
        predictions.append(prediction)

    # Report the whole dataset, then break it down by wording and lecture.
    groups = {"all": predictions}
    for prediction in predictions:
        for column, prefix in [("question_type", "type"), ("document", "document")]:
            group_name = f"{prefix}:{prediction[column]}"
            if group_name not in groups:
                groups[group_name] = []
            groups[group_name].append(prediction)

    metrics = {}
    for name, group in groups.items():
        result = calculate_accuracy(group)
        metrics[name] = result
        print(f"{name}: Top-1 {result['top1_accuracy']:.1%}, "
              f"Top-3 {result['top3_accuracy']:.1%} ({len(group)} questions)")

    results_folder = PROJECT_FOLDER / "results"
    results_folder.mkdir(exist_ok=True)
    predictions_file = results_folder / "tfidf_predictions.csv"
    with predictions_file.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)

    # Save the settings too, so we can reproduce this baseline later.
    settings = vectorizer.get_params()
    report = {
        "method": "word unigram TF-IDF with cosine similarity",
        "scikit_learn_version": sklearn.__version__,
        "n_chunks": len(chunks),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "vectorizer_parameters": {
            "lowercase": settings["lowercase"],
            "token_pattern": settings["token_pattern"],
            "ngram_range": settings["ngram_range"],
            "stop_words": settings["stop_words"],
            "smooth_idf": settings["smooth_idf"],
            "norm": settings["norm"],
            "sublinear_tf": settings["sublinear_tf"],
            "use_idf": settings["use_idf"],
        },
        "ranking": "positive scores only; exact ties resolved by chunk_id",
        "metrics": metrics,
    }
    metrics_file = results_folder / "tfidf_metrics.json"
    with metrics_file.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")

    print(f"\nSaved predictions and metrics in {results_folder}")


if __name__ == "__main__":
    main()
