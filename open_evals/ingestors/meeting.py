from typing import Any, cast, override

from datasets import (  # pyright: ignore[reportMissingTypeStubs]
    load_dataset,  # pyright: ignore[reportUnknownVariableType]
)
from tqdm.asyncio import tqdm

from open_evals.common import Document, QRel, Query
from open_evals.ingestors.common import BaseIngestor, clean_dataset


class MeetingIngestor(BaseIngestor):
    @override
    def dataset_id(self) -> str:
        return "evals/meeting"

    @override
    def ingest(self) -> tuple[list[Query], list[Document], list[QRel]]:
        dataset_name = "lytang/MeetingBank-transcript"
        dataset = cast(Any, load_dataset(dataset_name))
        dataset = dataset["train"]

        queries: list[Query] = []
        documents: list[Document] = []
        qrels: list[QRel] = []

        for i, item in tqdm(enumerate(dataset), desc="Datapoints"):
            queries.append(Query(id=f"q{i}", query=item["reference"]))
            documents.append(Document(id=f"d{i}", content=item["source"]))
            qrels.append(QRel(query_id=f"q{i}", document_id=f"d{i}", score=1))

        return clean_dataset(queries, documents, qrels)
