from typing import Any, cast, override

from datasets import (  # pyright: ignore[reportMissingTypeStubs]
    load_dataset,  # pyright: ignore[reportUnknownVariableType]
)
from tqdm import tqdm

from open_evals.common import Document, QRel, Query
from open_evals.ingestors.common import BaseIngestor, clean_dataset


class FiqaIngestor(BaseIngestor):
    @override
    def dataset_id(self) -> str:
        return "evals/fiqa"

    @override
    def ingest(self) -> tuple[list[Query], list[Document], list[QRel]]:
        dataset_name = "BeIR/fiqa"
        qrels_name = "BeIR/fiqa-qrels"
        # Load the datasets
        queries_dataset = cast(Any, load_dataset(dataset_name, "queries"))["queries"]
        corpus_dataset = cast(Any, load_dataset(dataset_name, "corpus"))["corpus"]

        # Load and concatenate qrels splits
        qrels_raw = cast(Any, load_dataset(qrels_name, "default"))
        qrels_dataset: list[Any] = []
        for split in ["validation", "test"]:
            qrels_dataset.extend(qrels_raw[split])

        # Create documents
        documents: list[Document] = []
        for document in tqdm(corpus_dataset, desc="Documents"):
            documents.append(
                Document(
                    id=document["_id"],
                    content=document["text"],
                )
            )

        # Create QRel objects
        test_and_valid_query_ids: set[str] = set()
        qrels: list[QRel] = []
        for item in tqdm(qrels_dataset, desc="QRels"):
            qrels.append(
                QRel(
                    query_id=str(item["query-id"]),
                    document_id=str(item["corpus-id"]),
                    score=float(item["score"]),
                )
            )
            test_and_valid_query_ids.add(qrels[-1].query_id)

        # Create Query objects
        queries: list[Query] = []
        for item in tqdm(queries_dataset, desc="Queries"):
            if item["_id"] not in test_and_valid_query_ids:
                continue
            queries.append(
                Query(
                    id=item["_id"],
                    query=item["text"],
                )
            )

        return clean_dataset(queries, documents, qrels)
