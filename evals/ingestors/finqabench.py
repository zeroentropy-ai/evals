from typing import Any, cast, override

from datasets import (  # pyright: ignore[reportMissingTypeStubs]
    load_dataset,  # pyright: ignore[reportUnknownVariableType]
)
from tqdm.asyncio import tqdm

from evals.common import Document, QRel, Query
from evals.ingestors.common import BaseIngestor, clean_dataset


class FinqabenchIngestor(BaseIngestor):
    @override
    def dataset_id(self) -> str:
        return "evals/finqabench"

    @override
    def ingest(self) -> tuple[list[Query], list[Document], list[QRel]]:
        dataset_name = "lighthouzai/finqabench"
        dataset = cast(Any, load_dataset(dataset_name))
        dataset = dataset["train"]

        # Create queries
        queries: list[Query] = []
        for i, question in tqdm(enumerate(dataset["Query"]), desc="Queries"):
            queries.append(Query(id=str(i), query=question, metadata={}))

        # Create documents
        documents: list[Document] = []
        for i, document in tqdm(enumerate(dataset["Context"]), desc="Documents"):
            documents.append(Document(id=str(i), content=document, metadata={}))

        # Create qrels
        qrels: list[QRel] = []
        for i in tqdm(range(len(dataset["Query"])), desc="QRels"):
            qrels.append(QRel(query_id=str(i), document_id=str(i), score=1.0))

        return clean_dataset(queries, documents, qrels)
