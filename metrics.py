def recall_at_k(predictions, ground_truth, k=3):
    """
    Computes Mean Recall@K
    """
    recalls = []
    for query_id, pred_list in predictions.items():
        top_k = pred_list[:k]
        relevant = set(ground_truth.get(query_id, []))
        if not relevant:
            continue
        hits = len(set(top_k) & relevant)
        recalls.append(hits / len(relevant))
    return sum(recalls) / len(recalls) if recalls else 0.0


def apk(actual, predicted, k=3):
    """
    Computes Average Precision@K for one query
    """
    if not actual:
        return 0.0

    actual_set = set(actual)
    predicted = predicted[:k]
    score = 0.0
    hits = 0

    for i, p in enumerate(predicted):
        if p in actual_set and p not in predicted[:i]:  # avoid duplicate hits
            hits += 1
            score += hits / (i + 1.0)

    return score / min(len(actual), k)


def mapk(predictions, ground_truth, k=3):
    """
    Computes Mean Average Precision@K
    """
    scores = []
    for query_id, pred_list in predictions.items():
        actual = ground_truth.get(query_id, [])
        scores.append(apk(actual, pred_list, k))
    return sum(scores) / len(scores) if scores else 0.0
