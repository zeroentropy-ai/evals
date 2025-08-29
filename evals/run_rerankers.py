import asyncio
import os
from collections import defaultdict
from contextlib import ExitStack
from typing import TextIO

from tqdm import tqdm

from evals.ai import (
    AIEmbeddingModel,
    AIRerankModel,
    AIModelAsReranker,
    ai_rerank,
    tiktoken_truncate_by_num_tokens,
)
from evals.common import (
    DocumentScores,
    QueryScores,
    RerankerName,
    RetrievalMethod,
    ZEDataset,
    ZEResults,
)
from evals.ingestors.common import BaseIngestor
from evals.types import (
    ALL_RERANKERS,
    DEFAULT_INCLUDE_RELEVANT_DOCS,
    DEFAULT_INGESTORS,
    DEFAULT_RERANKERS,
    DEFAULT_RETRIEVAL_METHOD,
)
from evals.utils import read_num_lines_pbar

NUM_SIMULTANEOUS_LINES = 25

RERANK_MAX_TOKENS = 4096
DEFAULT_MAX_BATCH_CHARACTERS = 750_000
company_to_max_batch_characters = {
    "modal": 100_000,
}


async def process_query(
    reranker: AIRerankModel | AIEmbeddingModel | AIModelAsReranker,
    query: str,
    documents: list[str],
) -> list[float]:
    # Handle AIModelAsReranker differently - it has its own rerank method
    if isinstance(reranker, AIModelAsReranker):
        return await reranker.rerank(query, documents)
    
    # For traditional rerankers, use batching
    max_batch_characters = company_to_max_batch_characters.get(
        reranker.company, DEFAULT_MAX_BATCH_CHARACTERS
    )

    batches: list[list[str]] = []
    cumulative_length = 0
    for document in documents:
        if len(batches) == 0 or cumulative_length > max_batch_characters:
            batches.append([])
            cumulative_length = 0
        batches[-1].append(document)
        cumulative_length += 20 + len(query.encode()) + len(document.encode())

    all_reranked_scores: list[float] = []
    for batch in batches:
        reranked_scores = await ai_rerank(
            reranker,
            query,
            batch,
        )
        all_reranked_scores.extend(reranked_scores)
    return all_reranked_scores


async def rerank_dataset(
    dataset: ZEDataset,
    rerankers: list[RerankerName],
    retrieval_method: RetrievalMethod,
    include_relevant_docs: bool,
) -> None:
    ze_results_path = dataset.ze_results_path(retrieval_method, include_relevant_docs)

    num_lines = read_num_lines_pbar(ze_results_path, display_name=dataset.id)

    processed_query_ids: dict[RerankerName, set[str]] = defaultdict(set)
    for reranker in rerankers:
        latest_ze_results_path = dataset.latest_ze_results_path(
            retrieval_method, include_relevant_docs, reranker
        )
        if os.path.exists(latest_ze_results_path):
            with open(latest_ze_results_path) as f:
                for line in f:
                    reranker_data = QueryScores.model_validate_json(line)
                    processed_query_ids[reranker].add(reranker_data.query_id)
        else:
            os.makedirs(os.path.dirname(latest_ze_results_path), exist_ok=True)

    pbar = tqdm(
        desc=f"Reranking {dataset.id}",
        total=num_lines,
    )
    pending_tasks: set[asyncio.Task[None]] = set()

    with open(ze_results_path) as f, ExitStack() as stack:
        f_write: dict[RerankerName, TextIO] = {
            reranker: stack.enter_context(
                open(
                    dataset.latest_ze_results_path(
                        retrieval_method, include_relevant_docs, reranker
                    ),
                    "a",
                )
            )
            for reranker in rerankers
        }

        async def wrapped_process_line(line: str) -> None:
            ze_results = ZEResults.model_validate_json(line)
            need_rerank: list[RerankerName] = []
            for reranker in rerankers:
                if ze_results.query_id not in processed_query_ids[reranker]:
                    need_rerank.append(reranker)
            if len(need_rerank) > 0:
                query_text = tiktoken_truncate_by_num_tokens(
                    ze_results.query, RERANK_MAX_TOKENS
                )
                document_texts = [
                    tiktoken_truncate_by_num_tokens(document.content, RERANK_MAX_TOKENS)
                    for document in ze_results.documents
                ]
                ground_truth_exists = any(
                    document.scores.get("human", 0) > 0
                    for document in ze_results.documents
                )
                if ground_truth_exists:
                    all_results: list[list[float]] = [
                        await process_query(
                            ALL_RERANKERS[reranker], query_text, document_texts
                        )
                        for reranker in need_rerank
                    ]
                else:
                    # NOTE: Skip reranker calls when there's no ground truth in the top
                    all_results = [
                        [-1 for _document_text in document_texts]
                        for _reranker in need_rerank
                    ]
                for reranker, results in zip(need_rerank, all_results, strict=False):
                    reranker_scores: QueryScores = QueryScores(
                        query_id=ze_results.query_id,
                        documents=[
                            DocumentScores(
                                document_id=document.id,
                                scores={
                                    "human": document.scores.get("human", 0),
                                    "reranker": result_score,
                                },
                            )
                            for document, result_score in zip(
                                ze_results.documents, results, strict=False
                            )
                        ],
                    )
                    f_write[reranker].write(reranker_scores.model_dump_json() + "\n")
                    f_write[reranker].flush()
            pbar.update(1)

        for line in f:
            task = asyncio.create_task(wrapped_process_line(line))
            pending_tasks.add(task)
            task.add_done_callback(pending_tasks.remove)

            while len(pending_tasks) >= NUM_SIMULTANEOUS_LINES:
                await asyncio.sleep(0.1)

        while len(pending_tasks) > 0:
            await asyncio.sleep(0.1)
    pbar.close()


async def run_rerankers(
    *,
    ingestors: list[BaseIngestor] = DEFAULT_INGESTORS,
    rerankers: list[RerankerName] = DEFAULT_RERANKERS,
    retrieval_method: RetrievalMethod = DEFAULT_RETRIEVAL_METHOD,
    include_relevant_docs: bool = DEFAULT_INCLUDE_RELEVANT_DOCS,
) -> None:
    datasets = [ingestor.dataset() for ingestor in ingestors]
    for i, dataset in enumerate(datasets):
        print(f"===> Reranking {dataset.id} (Dataset {i + 1}/{len(datasets)}) <===")
        await rerank_dataset(
            dataset,
            rerankers,
            retrieval_method,
            include_relevant_docs,
        )


if __name__ == "__main__":
    asyncio.run(run_rerankers())
