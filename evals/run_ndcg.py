import math
import os
from collections import defaultdict
from typing import cast

from evals.common import ZEDataset, ZEResults
from evals.run_embeddings import MERGE_STATUS, RETRIEVAL_METHOD
from evals.run_ingestors import EVAL_DATASETS
from evals.utils import argsort, avg

DATASETS = EVAL_DATASETS
K = cast(int | None, 10)
READ_NAME = f"{RETRIEVAL_METHOD}_{MERGE_STATUS}latest_ze_results.jsonl"

SHOW_DEFAULT = False


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


def analyze_ndcg(dataset: ZEDataset) -> None:
    print(f"NDCG@{K} for {dataset.id}:")

    ze_results_path = dataset.file_path(f"{RETRIEVAL_METHOD}_{MERGE_STATUS}ze_results.jsonl")
    if not os.path.exists(ze_results_path):
        print("- Missing File")
        return
    total_lines = 0
    with open(ze_results_path) as f:
        for _line in f:
            total_lines += 1

    all_ndcgs: dict[str, list[float]] = defaultdict(list)
    scored_ze_results_path = dataset.file_path(READ_NAME)
    if not os.path.exists(scored_ze_results_path):
        print("- Missing File")
        return
    scored_lines = 0
    with open(scored_ze_results_path) as f:
        for line in f:
            scored_lines += 1
            ze_results = ZEResults.model_validate_json(line)
            human_scores = [
                document.scores.get("human", 0.0) for document in ze_results.documents
            ]

            # Compute IDCG (ideal DCG)
            idcg = dcg(human_scores, human_scores, k=K)
            if idcg == 0:
                continue

            # Compute Default NDCG
            default_ndcg = (
                dcg(human_scores, [-i for i in range(len(human_scores))], k=K) / idcg
            )
            all_ndcgs["default"].append(default_ndcg)

            # Compute NDCG for all scores
            score_names = [
                score_name
                for score_name in ze_results.documents[0].scores.keys()
                if score_name != "human"
            ]
            for score_name in score_names:
                reranker_scores = [
                    document.scores[score_name] for document in ze_results.documents
                ]
                ndcg = dcg(human_scores, reranker_scores, k=K) / idcg
                all_ndcgs[score_name].append(ndcg)

    if scored_lines < total_lines:
        print(f"- {scored_lines}/{total_lines} are scored")
    for score_name, ndcgs in all_ndcgs.items():
        if score_name == "default" and not SHOW_DEFAULT:
            continue
        # assert len(ndcgs) == len(all_ndcgs["default"])
        if len(ndcgs) != len(all_ndcgs["default"]):
            print(f"{score_name}: {len(ndcgs)} / {len(all_ndcgs['default'])}")
        average_ndcg = avg(ndcgs)
        if len(ndcgs) > 1:
            stddev_ndcg = (
                sum((x - average_ndcg) ** 2 for x in ndcgs) / (len(ndcgs) - 1)
            ) ** 0.5
            stderr_ndcg = stddev_ndcg / (len(ndcgs) ** 0.5)
        else:
            stderr_ndcg = float("nan")
        print(f"- {score_name:.<25}{average_ndcg:.5f} ± {stderr_ndcg:.5f}")


def main() -> None:
    for dataset in EVAL_DATASETS:
        analyze_ndcg(dataset)


if __name__ == "__main__":
    main()
