from typing import Any, cast, override

from datasets import (  # pyright: ignore[reportMissingTypeStubs]
    load_dataset,  # pyright: ignore[reportUnknownVariableType]
)
from tqdm import tqdm

from evals.common import Document, QRel, Query
from evals.ingestors.common import BaseIngestor, clean_dataset


class StackoverflowqaIngestor(BaseIngestor):
    @override
    def dataset_id(self) -> str:
        return "evals/stackoverflowqa"

    @override
    def ingest(self) -> tuple[list[Query], list[Document], list[QRel]]:
        dataset_name = "mteb/stackoverflow-qa"

        # Load both question-answer pairs and the text corpus
        qa_dataset = cast(Any, load_dataset(dataset_name, "queries"))["queries"]
        corpus_dataset = cast(Any, load_dataset(dataset_name, "corpus"))["corpus"]

        # Create queries from questions
        queries: list[Query] = []
        for item in tqdm(qa_dataset, desc="Queries"):
            queries.append(
                Query(
                    id=item["_id"],
                    query=item["text"],
                    metadata={},
                )
            )

        # Create documents from the text corpus
        documents: list[Document] = []
        for doc in tqdm(corpus_dataset, desc="Documents"):
            documents.append(
                Document(
                    id=doc["_id"],
                    content=doc["text"],
                    metadata={},
                )
            )

        # Create qrels from relevant passage IDs
        qrels: list[QRel] = []
        for query, doc in tqdm(
            zip(qa_dataset, corpus_dataset, strict=True), desc="QRels"
        ):
            query_id = query["_id"]
            doc_id = doc["_id"]
            qrels.append(
                QRel(
                    query_id=query_id,
                    document_id=doc_id,
                    score=1.0,  # Binary relevance
                )
            )

        return clean_dataset(queries, documents, qrels)
