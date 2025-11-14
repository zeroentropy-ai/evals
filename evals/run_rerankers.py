import asyncio
from contextlib import ExitStack
from pathlib import Path
from typing import TextIO

import diskcache as dc  # pyright: ignore[reportMissingTypeStubs]
from tqdm import tqdm

from evals.ai import (
    AIEmbeddingModel,
    AIRerankModel,
    ai_rerank,
    tiktoken_truncate_by_num_tokens,
)
from evals.ai_rerank import (
    AIModelAsReranker,
    ai_rerank_by_ai_model,
)
from evals.common import (
    DocumentScores,
    QueryScores,
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
    RerankerName,
    RetrievalMethod,
)
from evals.utils import flatten, read_num_lines_pbar

USE_RERANKER_CACHE = True
NUM_SIMULTANEOUS_LINES = 25

RERANK_MAX_TOKENS = 4096
DEFAULT_MAX_BATCH_CHARACTERS = 750_000
company_to_max_batch_characters = {
    "modal": 100_000,
}
AI_MODEL_AS_RERANKER_MAX_BYTES = 50_000


async def process_query(
    reranker: AIRerankModel | AIEmbeddingModel | AIModelAsReranker,
    query: str,
    documents: list[str],
    # Cache
    cache: dc.Cache | None = None,
) -> list[float]:
    match reranker:
        case AIRerankModel() | AIEmbeddingModel():
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
            scores = flatten(
                await asyncio.gather(
                    *[
                        ai_rerank(
                            reranker,
                            query,
                            batch,
                            cache=cache,
                        )
                        for batch in batches
                    ]
                )
            )
        case AIModelAsReranker():
            scores = await ai_rerank_by_ai_model(
                reranker,
                query,
                documents,
                max_bytes=AI_MODEL_AS_RERANKER_MAX_BYTES,
                cache=cache,
            )
    return scores


async def rerank_dataset(
    dataset: ZEDataset,
    rerankers: list[RerankerName],
    retrieval_method: RetrievalMethod,
    include_relevant_docs: bool,
) -> None:
    ze_results_path = dataset.ze_results_path(retrieval_method, include_relevant_docs)

    # Create reranker caches
    reranker_caches: dict[RerankerName, dc.Cache | None] = {}
    for reranker in rerankers:
        reranker_caches[reranker] = (
            dc.Cache(
                directory=dataset.reranker_cache_path(reranker),
                eviction_policy="none",
            )
            if USE_RERANKER_CACHE
            else None
        )

    num_lines = read_num_lines_pbar(ze_results_path, display_name=dataset.id)

    pbar = tqdm(
        desc=f"Reranking {dataset.id}",
        total=num_lines,
    )
    pending_tasks: set[asyncio.Task[None]] = set()

    for reranker in rerankers:
        Path(
            dataset.ze_scores_path(retrieval_method, include_relevant_docs, reranker)
        ).parent.mkdir(parents=True, exist_ok=True)

    with open(ze_results_path) as f, ExitStack() as stack:
        f_write: dict[RerankerName, TextIO] = {
            reranker: stack.enter_context(
                open(
                    dataset.ze_scores_path(
                        retrieval_method, include_relevant_docs, reranker
                    ),
                    "w",
                )
            )
            for reranker in rerankers
        }

        async def wrapped_process_line(line: str) -> None:
            ze_results = ZEResults.model_validate_json(line)
            query_text = tiktoken_truncate_by_num_tokens(
                ze_results.query, RERANK_MAX_TOKENS
            )
            document_texts = [
                tiktoken_truncate_by_num_tokens(document.content, RERANK_MAX_TOKENS)
                for document in ze_results.documents
            ]
            ground_truth_exists = any(
                document.scores.get("human", 0) > 0 for document in ze_results.documents
            )
            if ground_truth_exists:
                all_results = await asyncio.gather(
                    *[
                        process_query(
                            ALL_RERANKERS[reranker],
                            query_text,
                            document_texts,
                            reranker_caches[reranker],
                        )
                        for reranker in rerankers
                    ]
                )
            else:
                # NOTE: Skip reranker calls when there's no ground truth in the top
                all_results = [
                    [-1 for _document_text in document_texts] for _reranker in rerankers
                ]
            for reranker, results in zip(rerankers, all_results, strict=False):
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

        def on_task_complete(task: asyncio.Task[None]) -> None:
            pending_tasks.remove(task)
            task.result()  # Throw exception if the task failed

        for line in f:
            task = asyncio.create_task(wrapped_process_line(line))
            pending_tasks.add(task)
            task.add_done_callback(on_task_complete)

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
