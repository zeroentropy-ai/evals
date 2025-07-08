from typing import Any, cast, override

from datasets import (  # pyright: ignore[reportMissingTypeStubs]
    load_dataset,  # pyright: ignore[reportUnknownVariableType]
)
from tqdm.asyncio import tqdm

from evals.common import Document, QRel, Query
from evals.ingestors.common import BaseIngestor, clean_dataset


class MbppIngestor(BaseIngestor):
    @override
    def dataset_id(self) -> str:
        return "evals/mbpp"

    @override
    def ingest(self) -> tuple[list[Query], list[Document], list[QRel]]:
        dataset_name = "Sing0402/mbpp"
        dataset = cast(Any, load_dataset(dataset_name))
        test_data = dataset["train"]

        queries: list[Query] = []
        documents: list[Document] = []
        qrels: list[QRel] = []

        document_id = 0
        for i in tqdm(range(len(test_data)), desc="Datapoints"):
            queries.append(Query(id=str(i), query=test_data["prompt"][i]))
            doc_list = test_data["test_list"][i]
            for doc in doc_list:
                documents.append(Document(id=str(document_id), content=doc))
                qrels.append(
                    QRel(query_id=str(i), document_id=str(document_id), score=1)
                )
                document_id += 1

        return clean_dataset(queries, documents, qrels)
