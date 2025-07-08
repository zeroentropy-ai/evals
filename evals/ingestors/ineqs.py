from typing import Any, cast, override

from datasets import (  # pyright: ignore[reportMissingTypeStubs]
    load_dataset,  # pyright: ignore[reportUnknownVariableType]
)
from tqdm.asyncio import tqdm

from evals.common import Document, QRel, Query
from evals.ingestors.common import BaseIngestor, clean_dataset


class IneqsIngestor(BaseIngestor):
    @override
    def dataset_id(self) -> str:
        return "evals/ineqs"

    @override
    def ingest(self) -> tuple[list[Query], list[Document], list[QRel]]:
        dataset_name = "AI4Math/IneqMath"
        dataset = cast(Any, load_dataset(dataset_name, split="train"))

        queries: list[Query] = []
        documents: list[Document] = []
        qrels: list[QRel] = []

        for datum in tqdm(dataset, desc="Datapoints"):
            queries.append(Query(id=datum["data_id"], query=datum["problem"]))
            documents.append(Document(id=datum["data_id"], content=datum["solution"]))
            qrels.append(
                QRel(query_id=datum["data_id"], document_id=datum["data_id"], score=1)
            )

        return clean_dataset(queries, documents, qrels)
