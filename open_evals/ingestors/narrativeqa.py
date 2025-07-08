from typing import Any, cast, override

from datasets import (  # pyright: ignore[reportMissingTypeStubs]
    load_dataset,  # pyright: ignore[reportUnknownVariableType]
)
from tqdm import tqdm

from open_evals.common import Document, QRel, Query
from open_evals.ingestors.common import BaseIngestor, clean_dataset


class NarrativeQAIngestor(BaseIngestor):
    @override
    def dataset_id(self) -> str:
        return "evals/narrativeqa"

    @override
    def ingest(self) -> tuple[list[Query], list[Document], list[QRel]]:
        dataset_name = "deepmind/narrativeqa"
        dataset = cast(Any, load_dataset(path=dataset_name))["train"]
        queries: list[Query] = []
        documents: list[Document] = []
        qrels: list[QRel] = []

        # This dataset is too big, will only load first 1000.
        for i, (query, document) in tqdm(enumerate(zip(dataset["question"][:1000], dataset["document"][:1000])), desc="Datapoints"):
            queries.append(Query(id=str(i), query=query["text"]))
            documents.append(
                Document(id=str(i), content=document["text"])
            )
            qrels.append(QRel(query_id=str(i), document_id=str(i), score=1))

        return clean_dataset(queries, documents, qrels)
