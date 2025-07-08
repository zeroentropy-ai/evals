from typing import Any, cast, override

from datasets import (  # pyright: ignore[reportMissingTypeStubs]
    load_dataset,  # pyright: ignore[reportUnknownVariableType]
)
from tqdm import tqdm

from evals.common import Document, QRel, Query
from evals.ingestors.common import BaseIngestor, clean_dataset


class BioasqIngestor(BaseIngestor):
    @override
    def dataset_id(self) -> str:
        return "evals/bioasq"

    @override
    def ingest(self) -> tuple[list[Query], list[Document], list[QRel]]:
        dataset_name = "kroshan/BioASQ"
        dataset = cast(Any, load_dataset(dataset_name, "default"))["train"]

        # Create queries from questions
        queries: list[Query] = []
        for index, item in tqdm(enumerate(dataset), desc="Queries"):
            queries.append(
                Query(
                    id=f"q_{index}",
                    query=item["question"],
                    metadata={},
                )
            )

        # Create documents from the text corpus
        documents: list[Document] = []
        for index, doc in tqdm(enumerate(dataset), desc="Documents"):
            documents.append(
                Document(
                    id=f"d_{index}",
                    content=doc["text"],
                    metadata={},
                )
            )

        # Create qrels from relevant passage IDs
        qrels: list[QRel] = []
        for index in tqdm(range(len(dataset)), desc="QRels"):
            query_id = f"q_{index}"
            doc_id = f"d_{index}"
            qrels.append(
                QRel(
                    query_id=query_id,
                    document_id=doc_id,
                    score=1.0,  # Binary relevance
                )
            )

        return clean_dataset(queries, documents, qrels)
