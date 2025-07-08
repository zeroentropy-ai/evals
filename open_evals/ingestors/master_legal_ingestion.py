from typing import Any, cast, override

from datasets import (  # pyright: ignore[reportMissingTypeStubs]
    load_dataset,  # pyright: ignore[reportUnknownVariableType]
)
from tqdm.asyncio import tqdm

from open_evals.common import Document, QRel, Query
from open_evals.ingestors.common import BaseIngestor, clean_dataset


class MasterLegalIngestor(BaseIngestor):
    _dataset_id: str
    dataset_name: str
    language: str | None
    split: str

    def __init__(
        self,
        dataset_id: str,
        *,
        dataset_name: str,
        language: str | None = None,
        split: str = "test",
    ) -> None:
        self._dataset_id = dataset_id
        self.dataset_name = dataset_name
        self.language = language
        self.split = split

    @override
    def dataset_id(self) -> str:
        return self._dataset_id

    @override
    def ingest(self) -> tuple[list[Query], list[Document], list[QRel]]:
        # Load the three subsets
        queries_dataset = cast(Any, load_dataset(self.dataset_name, "queries"))[
            "queries"
        ]
        corpus_dataset = cast(Any, load_dataset(self.dataset_name, "corpus"))["corpus"]
        qrels_dataset = cast(Any, load_dataset(self.dataset_name, "default"))[
            self.split  
        ]

        # Create Query objects
        queries: list[Query] = []
        for item in tqdm(queries_dataset, desc="Queries"):
            queries.append(
                Query(
                    id=item["_id"],
                    query=item["text"],
                )
            )

        # Create Document objects
        documents: list[Document] = []
        for item in tqdm(corpus_dataset, desc="Documents"):
            documents.append(
                Document(
                    id=item["_id"],
                    content=item["text"],
                )
            )

        # Create QRel objects
        qrels: list[QRel] = []
        for item in tqdm(qrels_dataset, desc="QRels"):
            qrels.append(
                QRel(
                    query_id=item["query-id"],
                    document_id=item["corpus-id"],
                    score=float(item["score"]),
                )
            )

        return clean_dataset(queries, documents, qrels)
