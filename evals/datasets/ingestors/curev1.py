from typing import Any, cast, override

from datasets import (  # pyright: ignore[reportMissingTypeStubs]
    concatenate_datasets,
    load_dataset,  # pyright: ignore[reportUnknownVariableType]
)
from tqdm import tqdm

from evals.datasets.common import Document, QRel, Query
from evals.datasets.ingestors.common import BaseIngestor, clean_dataset
from evals.utils import hash_str


class CureV1Ingestor(BaseIngestor):
    @override
    def dataset_id(self) -> str:
        return "evals/curev1"

    @override
    def ingest(self) -> tuple[list[Query], list[Document], list[QRel]]:
        dataset_name = "clinia/CUREv1"
        queries_dataset = cast(Any, load_dataset(dataset_name, "queries-en"))
        corpus_dataset = cast(Any, load_dataset(dataset_name, "corpus"))
        qrels_dataset = cast(Any, load_dataset(dataset_name, "qrels"))

        # Concatenate all sub-datasets in each category
        all_query_datasets = [queries_dataset[key] for key in queries_dataset.keys()]
        merged_queries = concatenate_datasets(all_query_datasets)
        all_corpus_datasets = [corpus_dataset[key] for key in corpus_dataset.keys()]
        merged_corpus = concatenate_datasets(all_corpus_datasets)
        all_qrels_datasets = [qrels_dataset[key] for key in qrels_dataset.keys()]
        merged_qrels = concatenate_datasets(all_qrels_datasets)

        # Extract objects
        queries: list[Query] = []
        for query in tqdm(merged_queries, desc="Queries"):
            queries.append(
                Query(
                    id=query["_id"],
                    query=query["text"],
                )
            )
        document_id_to_hashes: dict[str, str] = {}
        documents: list[Document] = []
        for document in tqdm(merged_corpus, desc="Documents"):
            document_hash = hash_str(document["text"])
            if document["_id"] in document_id_to_hashes:
                assert document_id_to_hashes[document["_id"]] == document_hash
                continue
            document_id_to_hashes[document["_id"]] = document_hash
            documents.append(
                Document(
                    id=document["_id"],
                    content=document["text"],
                )
            )
        qrels: list[QRel] = []
        for qrel in tqdm(merged_qrels, desc="QRels"):
            qrels.append(
                QRel(
                    query_id=qrel["query-id"],
                    document_id=qrel["corpus-id"],
                    score=qrel["score"],
                )
            )

        return clean_dataset(queries, documents, qrels)
