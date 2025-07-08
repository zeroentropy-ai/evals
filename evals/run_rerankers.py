import asyncio
import os

from tqdm import tqdm

from evals.ai import AIRerankModel, ai_rerank, tiktoken_truncate_by_num_tokens
from evals.common import ZEDataset, ZEResults
from evals.run_embeddings import MERGE_STATUS, RETRIEVAL_METHOD
from evals.run_ingestors import EVAL_DATASETS
from evals.utils import read_num_lines_pbar

# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false

DATASETS = EVAL_DATASETS
NUM_SIMULTANEOUS_LINES = 25
NUM_SIMULTANEOUS_RERANKS = 150
VOYAGE_RERANKER = AIRerankModel(company="voyageai", model="rerank-2")
RERANKERS: dict[str, AIRerankModel] = {
    # "cohere": AIRerankModel(company="cohere", model="rerank-v3.5"),
    # "salesforce": AIRerankModel(company="together", model="Salesforce/Llama-Rank-V1"),
    "zeroentropy-small": AIRerankModel(
        company="modal",
        model="https://npip99--ze-rerank-small-v0-3-0-model-endpoint.modal.run/",
    ),
    "zeroentropy-large": AIRerankModel(
        company="modal",
        model="https://npip99--ze-rerank-v0-3-0-model-endpoint.modal.run/",
    ),
    # "voyage": AIRerankModel(company="voyageai", model="rerank-2"),
}
RERANK_MAX_TOKENS = 4096
RERANK_MAX_BATCH_CHARACTERS = 64_000

SAVE_NAME = f"{RETRIEVAL_METHOD}_{MERGE_STATUS}latest_ze_results.jsonl"
READ_NAME = f"{RETRIEVAL_METHOD}_{MERGE_STATUS}ze_results.jsonl"

RERANK_SEMAPHORE = asyncio.Semaphore(NUM_SIMULTANEOUS_RERANKS)

DEBUG = False


async def process_query(
    reranker: AIRerankModel,
    query: str,
    documents: list[str],
) -> list[float]:
    batches: list[list[str]] = []
    cumulative_length = 0
    for document in documents:
        if len(batches) == 0 or cumulative_length > RERANK_MAX_BATCH_CHARACTERS:
            batches.append([])
            cumulative_length = 0
        batches[-1].append(document)
        cumulative_length += 20 + len(query) + len(document)

    all_reranked_scores: list[float] = []
    for batch in batches:
        reranked_scores = await ai_rerank(
            reranker,
            query,
            batch,
        )
        all_reranked_scores.extend(reranked_scores)
    return all_reranked_scores


async def rerank_ze_results(
    ze_results: ZEResults,
    rerankers: dict[str, AIRerankModel],
) -> ZEResults:
    ze_results = ze_results.model_copy(deep=True)

    query_text = tiktoken_truncate_by_num_tokens(ze_results.query, RERANK_MAX_TOKENS)
    document_texts = [
        tiktoken_truncate_by_num_tokens(document.content, RERANK_MAX_TOKENS)
        for document in ze_results.documents
    ]

    all_results = [
        await process_query(reranker, query_text, document_texts)
        for reranker in rerankers.values()
    ]

    for reranker_name, results in zip(rerankers.keys(), all_results, strict=False):
        for document, result in zip(ze_results.documents, results, strict=False):
            document.scores[reranker_name] = result

    return ze_results


async def rerank_dataset(
    dataset: ZEDataset,
    rerankers: dict[str, AIRerankModel],
) -> None:
    output_file_path = dataset.file_path(SAVE_NAME)
    input_file_path = dataset.file_path(READ_NAME)
    num_lines = read_num_lines_pbar(input_file_path, display_name=dataset.id)

    processed_query_ids: set[str] = set()
    if os.path.exists(output_file_path):
        with open(output_file_path) as f:
            for line in f:
                ze_results = ZEResults.model_validate_json(line)
                processed_query_ids.add(ze_results.query_id)

    pbar = tqdm(
        desc=f"Reranking {dataset.id}",
        total=num_lines,
    )
    pending_tasks: set[asyncio.Task[None]] = set()

    with open(input_file_path) as f, open(output_file_path, "a") as f_write:

        async def wrapped_process_line(line: str) -> None:
            ze_results = ZEResults.model_validate_json(line)
            if ze_results.query_id not in processed_query_ids:
                scored_ze_results = await rerank_ze_results(
                    ze_results,
                    rerankers,
                )
                f_write.write(scored_ze_results.model_dump_json() + "\n")
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


async def main() -> None:
    for i, dataset in enumerate(DATASETS):
        print(f"===> Reranking {dataset.id} (Dataset {i + 1}/{len(DATASETS)}) <===")
        await rerank_dataset(
            dataset,
            RERANKERS,
        )


if __name__ == "__main__":
    asyncio.run(main())
