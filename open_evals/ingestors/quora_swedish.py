from typing import Any, cast, override

from datasets import (  # pyright: ignore[reportMissingTypeStubs]
    load_dataset,  # pyright: ignore[reportUnknownVariableType]
    concatenate_datasets,
)
from tqdm.asyncio import tqdm

from open_evals.common import Document, QRel, Query
from open_evals.ingestors.common import BaseIngestor, clean_dataset


class QuoraSwedishIngestor(BaseIngestor):
    @override
    def dataset_id(self) -> str:
        return "evals/quora_swedish"
    
    @override
    def ingest(self) -> tuple[list[Query], list[Document], list[QRel]]:
        dataset_name = "Gabriel/quora_swe"

        dataset_test = cast(Any, load_dataset(path=dataset_name))["test"]
        dataset_valid = cast(Any, load_dataset(path=dataset_name))["validation"]
        dataset = concatenate_datasets([dataset_test, dataset_valid])

        queries: list[Query] = []
        documents: list[Document] = []
        qrels: list[QRel] = []

        for i, datum in tqdm(enumerate(dataset), desc="Datapoints"):
            queries.append(Query(id=str(i), query=datum["question1"]))
            documents.append(Document(id=str(i), content=datum["question2"]))
            qrels.append(QRel(query_id=str(i), document_id=str(i), score=1))

        return clean_dataset(queries, documents, qrels)