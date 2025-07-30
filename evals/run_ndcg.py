import math
import os

from evals.common import (
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

DEFAULT_K = 10

SHOW_DEFAULT = True


def dcg(
    ground_truth_scores: list[float],
    rerank_scores: list[float],
    *,
    k: int | None,
) -> float:
    # CAREFUL: -score instead of argsort[::-1] is used, so that stablesort is preferred
    reranked_order = argsort([-score for score in rerank_scores])
    dcg = sum(
        ground_truth_scores[idx] / math.log2(rank + 2)
        for rank, idx in enumerate(reranked_order[:k])
    )
    return dcg


def analyze_ndcg(
    dataset: ZEDataset,
    retrieval_method: RetrievalMethod,
    include_relevant_docs: bool,
    rerankers: list[RerankerName],
    k: int,
) -> None:
    print(f"NDCG@{k} for {dataset.id}:")

    ze_results_path = dataset.ze_results_path(retrieval_method, include_relevant_docs)
    if not os.path.exists(ze_results_path):
        print("- Missing ZeResults")
        return
    total_lines = 0
    ground_truth: dict[str, tuple[list[float], float]] = {}
    default_ndcgs: list[float] = []
    with open(ze_results_path) as f:
        for _line in f:
            total_lines += 1
            ze_results = ZEResults.model_validate_json(_line)
            human_scores: list[float] = [
                document.scores.get("human", 0.0) for document in ze_results.documents
            ]
            idcg = dcg(human_scores, human_scores, k=k)
            if idcg == 0:
                continue
            ground_truth[ze_results.query_id] = (human_scores, idcg)
            default_ndcgs.append(
                dcg(human_scores, [-i for i in range(len(human_scores))], k=k) / idcg
            )
    if SHOW_DEFAULT:
        average_ndcg = avg(default_ndcgs)
        if len(default_ndcgs) > 1:
            stderr_ndcg = (
                sum((x - average_ndcg) ** 2 for x in default_ndcgs)
                / (len(default_ndcgs) - 1)
            ) ** 0.5
            stderr_ndcg = stderr_ndcg / (len(default_ndcgs) ** 0.5)
        else:
            stderr_ndcg = float("nan")
        print(
            f"- {len(default_ndcgs)}/{total_lines} - {retrieval_method:.<25}{average_ndcg:.5f} ± {stderr_ndcg:.5f}"
        )

    for reranker in rerankers:
        latest_ze_results_path = dataset.latest_ze_results_path(
            retrieval_method, include_relevant_docs, reranker
        )
        if not os.path.exists(latest_ze_results_path):
            print(f"- Missing Latest ZeResults for {reranker}")
            continue
        all_ndcgs: list[float] = []
        with open(latest_ze_results_path) as f:
            for line in f:
                reranker_data = QueryScores.model_validate_json(line)
                if reranker_data.query_id not in ground_truth:
                    continue
                human_scores, idcg = ground_truth[reranker_data.query_id]
                reranker_scores = [
                    document.scores["reranker"] for document in reranker_data.documents
                ]
                ndcg = dcg(human_scores, reranker_scores, k=k) / idcg
                all_ndcgs.append(ndcg)
        if len(all_ndcgs) == 0:
            print(f"- No NDCG for {reranker}")
            continue
        average_ndcg = avg(all_ndcgs)
        if len(all_ndcgs) > 1:
            stddev_ndcg = (
                sum((x - average_ndcg) ** 2 for x in all_ndcgs) / (len(all_ndcgs) - 1)
            ) ** 0.5
            stderr_ndcg = stddev_ndcg / (len(all_ndcgs) ** 0.5)
        else:
            stderr_ndcg = float("nan")
        print(
            f"- {len(all_ndcgs)}/{total_lines} - {reranker:.<25}{average_ndcg:.5f} ± {stderr_ndcg:.5f}"
        )


def run_ndcg(
    *,
    ingestors: list[BaseIngestor] = DEFAULT_INGESTORS,
    retrieval_method: RetrievalMethod = DEFAULT_RETRIEVAL_METHOD,
    include_relevant_docs: bool = DEFAULT_INCLUDE_RELEVANT_DOCS,
    rerankers: list[RerankerName] = DEFAULT_RERANKERS,
    k: int = DEFAULT_K,
) -> None:
    datasets = [ingestor.dataset() for ingestor in ingestors]
    for dataset in datasets:
        analyze_ndcg(dataset, retrieval_method, include_relevant_docs, rerankers, k)


if __name__ == "__main__":
    run_ndcg()
