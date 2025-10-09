import os
from collections import defaultdict

from evals.common import (
    QRel,
    QueryScores,
    RerankerName,
    RetrievalMethod,
    ZEDataset,
    ZEResults,
)
from evals.ingestors.common import BaseIngestor
from evals.types import (
    DEFAULT_INCLUDE_RELEVANT_DOCS,
    DEFAULT_INGESTORS,
    DEFAULT_RERANKERS,
    DEFAULT_RETRIEVAL_METHOD,
)
from evals.utils import argsort, avg

DEFAULT_K = 20

SHOW_DEFAULT = True


def recall(
    ground_truth_scores: list[float],
    rerank_scores: list[float],
    *,
    k: int | None,
    total_relevant: float,
) -> float:
    # CAREFUL: -score instead of argsort[::-1] is used, so that stablesort is preferred
    reranked_order = argsort([-score for score in rerank_scores])

    # Count relevant documents (score > 0) in top k
    relevant_in_top_k = sum(
        1 for idx in reranked_order[:k] if ground_truth_scores[idx] > 0
    )

    # Calculate recall
    if total_relevant == 0:
        return 1
    return relevant_in_top_k / total_relevant


def analyze_recall(
    dataset: ZEDataset,
    retrieval_method: RetrievalMethod,
    include_relevant_docs: bool,
    rerankers: list[RerankerName],
    k: int,
) -> dict[RerankerName | RetrievalMethod, float]:
    print(f"Recall@{k} for {dataset.id}:")
    results: dict[RerankerName | RetrievalMethod, float] = {}

    # Load qrels to get total relevant documents per query
    qrels_path = dataset.qrels_path
    if not os.path.exists(qrels_path):
        print("- Missing qrels")
        return {}

    query_total_relevant: dict[str, float] = defaultdict(float)
    with open(qrels_path) as f:
        for line in f:
            qrel = QRel.model_validate_json(line)
            if qrel.score > 0:
                query_total_relevant[qrel.query_id] += qrel.score

    ze_results_path = dataset.ze_results_path(retrieval_method, include_relevant_docs)
    if not os.path.exists(ze_results_path):
        print("- Missing ZeResults")
        return {}
    total_lines = 0
    ground_truth: dict[str, list[float]] = {}
    default_recalls: list[float] = []
    with open(ze_results_path) as f:
        for _line in f:
            total_lines += 1
            ze_results = ZEResults.model_validate_json(_line)
            human_scores: list[float] = [
                document.scores.get("human", 0.0) for document in ze_results.documents
            ]
            ground_truth[ze_results.query_id] = human_scores
            total_relevant_for_query = query_total_relevant.get(ze_results.query_id, 0)
            default_recalls.append(
                recall(
                    human_scores,
                    [-i for i in range(len(human_scores))],
                    k=k,
                    total_relevant=total_relevant_for_query,
                )
            )
    if SHOW_DEFAULT:
        average_recall = avg(default_recalls)
        if len(default_recalls) > 1:
            stderr_recall = (
                sum((x - average_recall) ** 2 for x in default_recalls)
                / (len(default_recalls) - 1)
            ) ** 0.5
            stderr_recall = stderr_recall / (len(default_recalls) ** 0.5)
        else:
            stderr_recall = float("nan")
        results[retrieval_method] = average_recall
        print(
            f"- {len(default_recalls)}/{total_lines} - {retrieval_method:.<25}{average_recall:.5f} ± {stderr_recall:.5f}"
        )

    for reranker in rerankers:
        ze_scores_path = dataset.ze_scores_path(
            retrieval_method, include_relevant_docs, reranker
        )
        if not os.path.exists(ze_scores_path):
            print(f"- Missing reranker scores for {reranker}")
            continue
        all_recalls: list[float] = []
        with open(ze_scores_path) as f:
            for line in f:
                reranker_data = QueryScores.model_validate_json(line)
                if reranker_data.query_id not in ground_truth:
                    continue
                human_scores = ground_truth[reranker_data.query_id]
                reranker_scores = [
                    document.scores["reranker"] for document in reranker_data.documents
                ]
                total_relevant_for_query = query_total_relevant.get(
                    reranker_data.query_id, 0
                )
                recall_score = recall(
                    human_scores,
                    reranker_scores,
                    k=k,
                    total_relevant=total_relevant_for_query,
                )
                all_recalls.append(recall_score)
        if len(all_recalls) == 0:
            print(f"- No Recall for {reranker}")
            continue
        average_recall = avg(all_recalls)
        if len(all_recalls) > 1:
            stddev_recall = (
                sum((x - average_recall) ** 2 for x in all_recalls)
                / (len(all_recalls) - 1)
            ) ** 0.5
            stderr_recall = stddev_recall / (len(all_recalls) ** 0.5)
        else:
            stderr_recall = float("nan")
        results[reranker] = average_recall
        print(
            f"- {len(all_recalls)}/{total_lines} - {reranker:.<25}{average_recall:.5f} ± {stderr_recall:.5f}"
        )

    return results


def run_recall(
    *,
    ingestors: list[BaseIngestor] = DEFAULT_INGESTORS,
    retrieval_method: RetrievalMethod = DEFAULT_RETRIEVAL_METHOD,
    include_relevant_docs: bool = DEFAULT_INCLUDE_RELEVANT_DOCS,
    rerankers: list[RerankerName] = DEFAULT_RERANKERS,
    k: int = DEFAULT_K,
) -> None:
    datasets = [ingestor.dataset() for ingestor in ingestors]
    reranker_to_recalls: dict[RerankerName | RetrievalMethod, list[float]] = (
        defaultdict(list)
    )
    for dataset in datasets:
        results = analyze_recall(
            dataset, retrieval_method, include_relevant_docs, rerankers, k
        )
        for reranker, recall in results.items():
            reranker_to_recalls[reranker].append(recall)

    print(f"Recall@{k} (Avg) For all datasets")
    for reranker, recalls in reranker_to_recalls.items():
        average_recall = avg(recalls)
        print(f"- {reranker:.<25}{average_recall:.5f}")


if __name__ == "__main__":
    run_recall()
