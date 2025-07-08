from typing import Any, cast, override

from datasets import (  # pyright: ignore[reportMissingTypeStubs]
    load_dataset,  # pyright: ignore[reportUnknownVariableType]
)
from tqdm import tqdm

from evals.common import Document, QRel, Query
from evals.ingestors.common import BaseIngestor, clean_dataset


class LeetcodeMultiLanguageIngestor(BaseIngestor):
    def __init__(self, language: str) -> None:
        self.language: str = language

    @override
    def dataset_id(self) -> str:
        return f"evals/leetcode_{self.language}"
    
    @override
    def ingest(self) -> tuple[list[Query], list[Document], list[QRel]]:
        dataset_name = "greengerong/leetcode"
        dataset = cast(Any, load_dataset(path=dataset_name))["train"]
        queries: list[Query] = []
        documents: list[Document] = []
        qrels: list[QRel] = []

        for datum in tqdm(dataset, desc="Datapoints"):
            queries.append(Query(id=str(datum["id"]), query=datum["content"]))
            documents.append(Document(id=str(datum["id"]), content=datum[self.language]))
            qrels.append(QRel(query_id=str(datum["id"]), document_id=str(datum["id"]), score=1))
        
        return clean_dataset(queries, documents, qrels)