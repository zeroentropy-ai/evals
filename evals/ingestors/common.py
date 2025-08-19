import math
from abc import ABC, abstractmethod
from random import Random
from typing import final

import tiktoken
from tqdm.asyncio import tqdm

from evals.common import Document, QRel, Query, ZEDataset
from evals.utils import hash_str


class BaseIngestor(ABC):
    @abstractmethod
    def dataset_id(self) -> str: ...

    @abstractmethod
    def ingest(self) -> tuple[list[Query], list[Document], list[QRel]]: ...

    @final
    def dataset(self) -> ZEDataset:
        return ZEDataset(id=self.dataset_id())


def remove_empty_queries(
    queries: list[Query],
    documents: list[Document],
    qrels: list[QRel],
) -> tuple[list[Query], list[Document], list[QRel]]:
    queries = [query for query in queries if len(query.query.strip()) > 0]
    query_ids = set(query.id for query in queries)
    qrels = [qrel for qrel in qrels if qrel.query_id in query_ids]
    return (queries, documents, qrels)


def remove_nonpositive_queries(
    queries: list[Query],
    documents: list[Document],
    qrels: list[QRel],
) -> tuple[list[Query], list[Document], list[QRel]]:
    query_id_with_positive: set[str] = set()
    for qrel in qrels:
        if qrel.score > 0:
            query_id_with_positive.add(qrel.query_id)

    queries = [query for query in queries if query.id in query_id_with_positive]

    qrels = [qrel for qrel in qrels if qrel.query_id in query_id_with_positive]

    return (queries, documents, qrels)


def limit_queries(
    queries: list[Query],
    documents: list[Document],
    qrels: list[QRel],
    *,
    limit: int | None,
    seed: str,
) -> tuple[list[Query], list[Document], list[QRel]]:
    rng = Random()
    rng.seed(seed)

    queries = queries[:]
    rng.shuffle(queries)
    queries = queries[:limit]

    included_query_ids = set(query.id for query in queries)

    qrels = [qrel for qrel in qrels if qrel.query_id in included_query_ids]

    return (queries, documents, qrels)


# The input could be invalid (duplicate pkeys, a qrel fkey that doesn't exist).
# This fixes that
def validate_dataset(
    queries: list[Query],
    documents: list[Document],
    qrels: list[QRel],
) -> tuple[list[Query], list[Document], list[QRel]]:
    query_ids = {query.id for query in queries}
    document_ids = {document.id for document in documents}
    qrel_ids = {(qrel.query_id, qrel.document_id) for qrel in qrels}
    assert len(query_ids) == len(queries)
    assert len(document_ids) == len(documents)
    assert len(qrel_ids) == len(qrels)

    qrels_new: list[QRel] = []
    for qrel in qrels:
        if (qrel.query_id in query_ids) and (qrel.document_id in document_ids):
            qrels_new.append(qrel)

    return queries, documents, qrels_new


def remove_duplicates(
    queries: list[Query],
    documents: list[Document],
    qrels: list[QRel],
) -> tuple[list[Query], list[Document], list[QRel]]:
    def remove_duplicates_by_dedup_keys[T](
        ids: list[str], items: list[T], dedup_keys: list[str]
    ) -> tuple[list[T], dict[str, str]]:
        new_items: list[T] = []
        key_to_representative_id: dict[str, str] = {}
        mapping: dict[str, str] = {}
        for id, item, key in zip(ids, items, dedup_keys, strict=True):
            if key not in key_to_representative_id:
                key_to_representative_id[key] = id
                new_items.append(item)
            mapping[id] = key_to_representative_id[key]
        return (new_items, mapping)

    query_ids = [query.id for query in queries]
    document_ids = [document.id for document in documents]
    query_hashes = [hash_str(query.query) for query in queries]
    document_hashes = [hash_str(document.format_string()) for document in documents]
    queries, query_mapping = remove_duplicates_by_dedup_keys(
        query_ids, queries, query_hashes
    )
    documents, document_mapping = remove_duplicates_by_dedup_keys(
        document_ids, documents, document_hashes
    )

    existing_qrels: dict[tuple[str, str], QRel] = {}
    for qrel in qrels:
        qrel.query_id = query_mapping[qrel.query_id]
        qrel.document_id = document_mapping[qrel.document_id]
        key = (qrel.query_id, qrel.document_id)
        if key in existing_qrels:
            existing_qrels[key].score = max(existing_qrels[key].score, qrel.score)
            continue
        else:
            existing_qrels[key] = qrel
    qrels = list(existing_qrels.values())

    return (queries, documents, qrels)


encoding = tiktoken.get_encoding("cl100k_base")


def split_by_tokens(s: str, max_tokens: int) -> list[str]:
    if len(s.encode("utf-8")) <= max_tokens:  # Tokenization is always compressive
        return [s]

    tokens = encoding.encode(s)
    total_tokens = len(tokens)

    # Determine minimum number of chunks needed
    num_chunks = math.ceil(total_tokens / max_tokens)

    # Compute chunk size for even distribution
    base_chunk_size = math.ceil(total_tokens / num_chunks)

    chunks: list[str] = []
    for i in range(num_chunks):
        start = i * base_chunk_size
        end = min((i + 1) * base_chunk_size, total_tokens)
        chunk_tokens = tokens[start:end]
        chunk_str = encoding.decode(chunk_tokens)
        chunks.append(chunk_str)

    return chunks


def chunk_long_strings(
    queries: list[Query],
    documents: list[Document],
    qrels: list[QRel],
    *,
    max_tokens: int = 4096,
) -> tuple[list[Query], list[Document], list[QRel]]:
    num_truncated_queries = 0
    for query in queries:
        chunks = split_by_tokens(query.query, max_tokens)
        if len(chunks) > 1:
            num_truncated_queries += 1
        query.query = chunks[0]
    if num_truncated_queries > 0:
        print(f"Truncated {num_truncated_queries}/{len(queries)} queries.")

    new_documents: list[Document] = []
    chunked_document_ids: dict[str, list[str]] = {}
    for document in tqdm(documents, desc="Chunking Documents"):
        chunks = split_by_tokens(document.content, max_tokens)
        if len(chunks) > 1:
            chunked_document_ids[document.id] = []
            for i, chunk in enumerate(chunks):
                chunk_document_id = f"{document.id}-chunk-{i}"
                chunked_document_ids[document.id].append(f"{document.id}-chunk-{i}")
                new_documents.append(
                    Document(
                        id=f"{document.id}-chunk-{i}",
                        content=chunk,
                        metadata={
                            **document.metadata,
                            "groupby": document.id,
                        },
                    )
                )
        else:
            new_documents.append(document)

    new_qrels: list[QRel] = []
    for qrel in qrels:
        if qrel.document_id in chunked_document_ids:
            for chunk_document_id in chunked_document_ids[qrel.document_id]:
                new_qrels.append(
                    QRel(
                        query_id=qrel.query_id,
                        document_id=chunk_document_id,
                        score=qrel.score,
                    )
                )
        else:
            new_qrels.append(qrel)

    if len(documents) != len(new_documents):
        print(
            f"Chunking modified len(documents): {len(documents)} -> {len(new_documents)}"
        )

    return (queries, new_documents, new_qrels)


def clean_dataset(
    queries: list[Query],
    documents: list[Document],
    qrels: list[QRel],
) -> tuple[list[Query], list[Document], list[QRel]]:
    # Validate no duplicate IDs, and remove invalid qrels
    queries, documents, qrels = validate_dataset(queries, documents, qrels)

    # Clean the string content
    queries, documents, qrels = chunk_long_strings(
        *remove_nonpositive_queries(
            *remove_duplicates(
                *remove_empty_queries(queries, documents, qrels),
            )
        )
    )
    assert len(queries) > 0 and len(qrels) > 0

    return queries, documents, qrels
