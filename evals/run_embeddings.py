import asyncio
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, cast

import diskcache as dc  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
from rank_bm25 import BM25Okapi  # pyright: ignore[reportMissingTypeStubs]
from tqdm import tqdm

from evals.ai import (
    AIEmbeddingModel,
    AIEmbeddingType,
    ai_embedding,
    tiktoken_truncate_by_num_tokens,
)
from evals.common import Document, QRel, Query, ZEDataset, ZEResults
from evals.run_ingestors import EVAL_DATASETS
from evals.utils import ROOT

DATASETS = EVAL_DATASETS
USE_EMBEDDINGS_CACHE = True
EMBEDDING_MAX_TOKENS = 8192

# Configuration: Change this constant to select retrieval method
RETRIEVAL_METHOD: Literal["openai_small", "bm25", "hybrid"] = "bm25"
INCLUDE_RELEVANT_DOCS = False
MERGE_STATUS = "merged_" if INCLUDE_RELEVANT_DOCS else ""
SAVE_NAME = f"{RETRIEVAL_METHOD}_{MERGE_STATUS}ze_results.jsonl"

type nparr = np.ndarray[Any, Any]


async def get_openai_small_embeddings(
    queries: list[Query],
    documents: list[Document],
    k: int,
    embeddings_cache: dc.Cache | None = None,
) -> tuple[nparr, nparr]:
    """Calculate OpenAI embedding similarity scores between queries and documents."""
    embedding_model = AIEmbeddingModel(company="openai", model="text-embedding-3-small")
    # Calculate query embeddings
    pbar = tqdm(
        desc="Query Embeddings",
        total=len(queries),
    )
    query_embeddings = np.array(
        await ai_embedding(
            model=embedding_model,
            texts=[
                tiktoken_truncate_by_num_tokens(query.query, EMBEDDING_MAX_TOKENS)
                for query in queries
            ],
            embedding_type=AIEmbeddingType.QUERY,
            cache=embeddings_cache,
            callback=lambda: pbar.update(1),
        )
    )
    pbar.close()
    document_texts = [
        tiktoken_truncate_by_num_tokens(document.content, EMBEDDING_MAX_TOKENS)
        for document in tqdm(documents, desc="Truncating Documents")
    ]

    # Calculate dot products
    DOT_PRODUCT_BATCH_SIZE = 32000
    num_queries = len(queries)
    num_docs = len(documents)
    k = min(k, num_docs)

    dot_products = np.empty((num_queries, num_docs), dtype=np.float32)

    pbar = tqdm(
        desc="Document Embeddings",
        total=len(documents),
    )
    for j in range(0, num_docs, DOT_PRODUCT_BATCH_SIZE):
        pbar.set_postfix_str("Running Embeddings")
        end = min(j + DOT_PRODUCT_BATCH_SIZE, num_docs)
        # Compute dot product for all queries against a batch of documents
        document_embeddings = np.array(
            await ai_embedding(
                model=embedding_model,
                texts=document_texts[j:end],
                embedding_type=AIEmbeddingType.DOCUMENT,
                cache=embeddings_cache,
                callback=lambda: pbar.update(1),
            )
        )
        pbar.set_postfix_str("Calculating Dot Product")
        dot_products[:, j:end] = query_embeddings @ document_embeddings.T
    pbar.set_postfix_str()
    pbar.close()

    # Get top k documents for each query
    print(f"Finding top {k} documents...")
    top_indices = np.argpartition(dot_products, -k, axis=1)[:, -k:]  # (num_queries, k)
    sorted_top_indices = np.argsort(
        np.take_along_axis(dot_products, top_indices, axis=1), axis=1
    )[:, ::-1]  # (num_queries, k)
    top_sorted_indices = np.take_along_axis(top_indices, sorted_top_indices, axis=1)

    # Get the actual scores for the top k documents
    top_scores = np.take_along_axis(dot_products, top_sorted_indices, axis=1)

    return top_sorted_indices, top_scores


def get_bm25_embeddings(
    queries: list[Query], documents: list[Document], k: int
) -> tuple[nparr, nparr]:
    """Calculate BM25 similarity scores between queries and documents."""
    # Prepare document texts
    document_texts = [
        tiktoken_truncate_by_num_tokens(document.content, EMBEDDING_MAX_TOKENS)
        for document in tqdm(documents, desc="Truncating Documents")
    ]

    # Tokenize documents for BM25
    print("Tokenizing documents for BM25...")
    tokenized_docs = [
        doc.lower().split() for doc in tqdm(document_texts, desc="Tokenizing")
    ]

    # Initialize BM25
    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized_docs)

    # Calculate BM25 scores for all queries
    num_queries = len(queries)
    num_docs = len(documents)
    k = min(k, num_docs)
    similarity_scores = np.empty((num_queries, num_docs), dtype=np.float32)

    for i, query in enumerate(tqdm(queries, desc="BM25 Scoring")):
        query_text = tiktoken_truncate_by_num_tokens(query.query, EMBEDDING_MAX_TOKENS)
        tokenized_query = query_text.lower().split()
        scores = cast(Any, bm25.get_scores(tokenized_query))  # pyright: ignore[reportUnknownMemberType]
        similarity_scores[i] = scores

    # Get top k documents for each query
    print(f"Finding top {k} documents...")
    top_indices = np.argpartition(similarity_scores, -k, axis=1)[
        :, -k:
    ]  # (num_queries, k)
    sorted_top_indices = np.argsort(
        np.take_along_axis(similarity_scores, top_indices, axis=1), axis=1
    )[:, ::-1]  # (num_queries, k)
    top_sorted_indices = np.take_along_axis(top_indices, sorted_top_indices, axis=1)

    # Get the actual scores for the top k documents
    top_scores = np.take_along_axis(similarity_scores, top_sorted_indices, axis=1)

    return top_sorted_indices, top_scores


async def get_hybrid_embeddings(
    queries: list[Query],
    documents: list[Document],
    k: int,
    embeddings_cache: dc.Cache | None = None,
) -> tuple[nparr, nparr]:
    embedding_indices, embedding_scores = await get_openai_small_embeddings(
        queries, documents, 3 * k, embeddings_cache
    )

    bm25_indices, bm25_scores = get_bm25_embeddings(queries, documents, 3 * k)

    num_queries = len(queries)
    num_docs = len(documents)
    k = min(k, num_docs)

    final_indices: list[Any] = []
    final_scores: list[Any] = []

    for query_idx in range(num_queries):
        embedding_doc_scores: dict[Any, Any] = {}
        bm25_doc_scores: dict[Any, Any] = {}

        # Add embedding scores
        for i, doc_idx in enumerate(embedding_indices[query_idx]):
            embedding_doc_scores[doc_idx] = embedding_scores[query_idx][i]
        for i, doc_idx in enumerate(bm25_indices[query_idx]):
            bm25_doc_scores[doc_idx] = bm25_scores[query_idx][i]

        sorted_embedding_tuples = sorted(
            embedding_doc_scores.items(), key=lambda x: x[1], reverse=True
        )
        sorted_bm25_tuples = sorted(
            bm25_doc_scores.items(), key=lambda x: x[1], reverse=True
        )

        doc_scores: dict[Any, Any] = {}
        for i in range(len(sorted_embedding_tuples)):
            doc_scores[sorted_embedding_tuples[i][0]] = 1 / (i + 1)
        for i in range(len(sorted_bm25_tuples)):
            doc_scores[sorted_bm25_tuples[i][0]] = doc_scores.get(
                sorted_bm25_tuples[i][0], 0
            ) + (1 / (i + 1))

        # Sort by score and take top k
        sorted_docs: list[tuple[Any, Any]] = sorted(
            doc_scores.items(), key=lambda x: x[1], reverse=True
        )[:k]

        final_indices.append([doc_idx for doc_idx, _ in sorted_docs])
        final_scores.append([score for _, score in sorted_docs])

    return np.array(final_indices, dtype=object), np.array(final_scores, dtype=object)


async def generate_embeddings(
    dataset: ZEDataset,
    *,
    k: int = 100,
    embedding_model: AIEmbeddingModel | None = None,
) -> None:
    rng = random.Random()
    rng.seed(dataset.id)

    # Load the dataset
    print("Loading dataset...")
    with open(dataset.queries_path) as f:
        queries: list[Query] = [Query.model_validate_json(line) for line in f]
    with open(dataset.documents_path) as f:
        documents = [Document.model_validate_json(line) for line in f]
    with open(dataset.qrels_path) as f:
        qrels = [QRel.model_validate_json(line) for line in f]
    assert len(queries) <= 10_000, (
        "We don't support dot product with too many queries yet."
    )

    # Load the qrels, but map to indexes
    document_id_to_index = {document.id: i for i, document in enumerate(documents)}
    query_and_document_id_to_qrel: dict[tuple[str, str], QRel] = {}
    query_id_to_qrel_indices: dict[str, list[int]] = defaultdict(list)
    for qrel in qrels:
        if qrel.score < 0.01:
            continue
        assert (qrel.query_id, qrel.document_id) not in query_and_document_id_to_qrel
        query_and_document_id_to_qrel[(qrel.query_id, qrel.document_id)] = qrel
        query_id_to_qrel_indices[qrel.query_id].append(
            document_id_to_index[qrel.document_id]
        )

    # Calculate similarity scores using selected retrieval method
    embeddings_cache = (
        dc.Cache(
            dataset.file_path(f"{MERGE_STATUS}embeddings_cache.db"),
            eviction_policy="none",
        )
        if USE_EMBEDDINGS_CACHE
        else None
    )

    print(f"Using retrieval method: {RETRIEVAL_METHOD}")

    if RETRIEVAL_METHOD == "openai_small":
        top_sorted_indices, similarity_scores = await get_openai_small_embeddings(
            queries, documents, k, embeddings_cache
        )
    elif RETRIEVAL_METHOD == "bm25":
        top_sorted_indices, similarity_scores = get_bm25_embeddings(
            queries, documents, k
        )
    elif RETRIEVAL_METHOD == "hybrid":
        top_sorted_indices, similarity_scores = await get_hybrid_embeddings(
            queries, documents, k, embeddings_cache
        )

    # Save all necessary data
    with open(dataset.file_path(SAVE_NAME), "w") as f:
        queries_processed = 0
        queries_skipped = 0

        for query_index, query in enumerate(queries):
            qrel_indices = query_id_to_qrel_indices[query.id]
            query_top_sorted_indices = top_sorted_indices[query_index]
            query_similarity_scores = similarity_scores[query_index]

            # Convert to lists if they're numpy arrays
            if hasattr(query_top_sorted_indices, "tolist"):
                query_top_sorted_indices = query_top_sorted_indices.tolist()
            if hasattr(query_similarity_scores, "tolist"):
                query_similarity_scores = query_similarity_scores.tolist()

            if INCLUDE_RELEVANT_DOCS:
                # Force include all relevant documents
                if len(qrel_indices) > k:
                    # If more relevant docs than k, take only first k relevant docs
                    final_indices = qrel_indices[:k]
                    final_scores = [
                        0.0
                    ] * k  # Will be set below based on method results
                else:
                    # Get natural results, remove duplicates with relevant docs, add relevant docs
                    natural_indices = [
                        i for i in query_top_sorted_indices if i not in qrel_indices
                    ]
                    qty_to_remove = len(natural_indices) + len(qrel_indices) - k
                    if qty_to_remove > 0:
                        natural_indices = natural_indices[:-qty_to_remove]
                    final_indices = natural_indices + qrel_indices
                    final_scores = [0.0] * len(final_indices)  # Will be set below

                # Create score lookup for final ranking
                score_lookup: dict[Any, Any] = {}
                for idx, score in zip(
                    query_top_sorted_indices, query_similarity_scores, strict=True
                ):
                    score_lookup[idx] = score

                # Set scores and sort by similarity
                for i, idx in enumerate(final_indices):
                    final_scores[i] = score_lookup.get(idx, 0.0)

                # Sort by score (descending)
                sorted_pairs = sorted(
                    zip(final_indices, final_scores, strict=True),
                    key=lambda x: x[1],
                    reverse=True,
                )
                query_top_sorted_indices = [idx for idx, _ in sorted_pairs]
                query_similarity_scores = [score for _, score in sorted_pairs]

            else:
                # Natural top-k only, skip queries with no relevant docs
                relevant_in_top_k = [
                    i for i in query_top_sorted_indices if i in qrel_indices
                ]

                if not relevant_in_top_k:
                    # Skip this query as it has no relevant documents in top k
                    queries_skipped += 1
                    continue

            documents_top: list[Document] = []
            for i, index in enumerate(query_top_sorted_indices):
                scores = {
                    f"{RETRIEVAL_METHOD}": float(query_similarity_scores[i]),
                }
                qrel = query_and_document_id_to_qrel.get(
                    (query.id, documents[index].id), None
                )
                if qrel is not None:
                    scores["human"] = qrel.score
                documents_top.append(
                    Document(
                        id=documents[index].id,
                        content=documents[index].content,
                        scores=scores,
                    )
                )

            ze_results = ZEResults(
                query_id=query.id,
                query=query.query,
                documents=documents_top,
            )
            f.write(ze_results.model_dump_json() + "\n")
            queries_processed += 1

    print(f"Data saved to {Path(dataset.ze_results_path).relative_to(ROOT)}")
    print(f"Processed {queries_processed} queries")
    if queries_skipped > 0:
        print(
            f"Skipped {queries_skipped} queries with no relevant documents in top {k}"
        )


async def main() -> None:
    for i, dataset in enumerate(DATASETS):
        print(f"===> Embedding {dataset.id} (Dataset {i + 1}/{len(DATASETS)}) <===")
        await generate_embeddings(dataset)


if __name__ == "__main__":
    asyncio.run(main())
