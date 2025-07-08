from typing import Any, cast, override

from datasets import (  # pyright: ignore[reportMissingTypeStubs]
    load_dataset,  # pyright: ignore[reportUnknownVariableType]
)
from tqdm.asyncio import tqdm

from open_evals.common import Document, QRel, Query
from open_evals.ingestors.common import BaseIngestor, clean_dataset


class PandasIngestor(BaseIngestor):
    @override
    def dataset_id(self) -> str:
        return "evals/pandas"
    
    @override
    def ingest(self) -> tuple[list[Query], list[Document], list[QRel]]:
        dataset_name = "pacovaldez/pandas-documentation"
        dataset = cast(Any, load_dataset(path=dataset_name))["train"]

        queries: list[Query] = []
        documents: list[Document] = []
        qrels: list[QRel] = []

        for i, datum in tqdm(enumerate(dataset), desc="Datapoints"):
            queries.append(Query(id=str(i), query=datum["title"]))
            documents.append(Document(id=str(i), content=datum["context"]))
            qrels.append(QRel(query_id=str(i), document_id=str(i), score=1))
        
        return clean_dataset(queries, documents, qrels)