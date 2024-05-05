from typing import List, Dict

import numpy as np

def avg_prec(correct_duplicates: List, retrieved_duplicates: List) -> float:
    """
    Get average precision(AP) for a single query given correct and retrieved file names.

    Args:
        correct_duplicates: List of correct duplicates i.e., ground truth)
        retrieved_duplicates: List of retrieved duplicates for one single query

    Returns:
        Average precision for this query.
    """
    if len(retrieved_duplicates) == 0 and len(correct_duplicates) == 0:
        return 1.0

    if not len(retrieved_duplicates) or not len(correct_duplicates):
        return 0.0

    count_real_correct = len(correct_duplicates)
    relevance = np.array([1 if i in correct_duplicates else 0 for i in retrieved_duplicates])
    relevance_cumsum = np.cumsum(relevance)
    prec_k = [relevance_cumsum[k] / (k + 1) for k in range(len(relevance))]
    prec_and_relevance = [relevance[k] * prec_k[k] for k in range(len(relevance))]
    avg_precision = np.sum(prec_and_relevance) / count_real_correct
    return avg_precision


def mean_metric(ground_truth: Dict, retrieved: Dict) -> float:
    """
    Get mean of specified metric.

    Args:
        metric_func: metric function on which mean is to be calculated across all queries

    Returns:
        float representing mean of the metric across all queries
    """
    metric_func = avg_prec
    metric_vals = []

    for k in ground_truth.keys():
        metric_vals.append(metric_func(ground_truth[k], retrieved[k]))
    return np.mean(metric_vals)