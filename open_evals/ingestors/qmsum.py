from typing import Any, cast, override

from datasets import (  # pyright: ignore[reportMissingTypeStubs]
    load_dataset,  # pyright: ignore[reportUnknownVariableType]
    concatenate_datasets,
)
from tqdm.asyncio import tqdm

from open_evals.common import Document, QRel, Query
from open_evals.ingestors.common import BaseIngestor


class QMSumIngestor(BaseIngestor):
    @override
    def dataset_id(self) -> str:
        return "evals/qmsum"

    @override
    def ingest(self) -> tuple[list[Query], list[Document], list[QRel]]:
        dataset_name = "pszemraj/qmsum-cleaned"
        dataset = cast(Any, load_dataset(dataset_name, "default"))
        test_data = concatenate_datasets([dataset["train"], dataset["validation"], dataset["test"]])


        queries: list[Query] = []
        documents: list[Document] = []
        qrels: list[QRel] = []

        document_id = 0
        all_queries = test_data["output"] #surprisingly. The output is much shorter than the input "document"
        all_query_ids = test_data["id"]
        all_doc_info = test_data["input"]
        for query, query_id, doc in tqdm(zip(all_queries, all_query_ids, all_doc_info), desc="Datapoints"):
            queries.append(Query(id=str(query_id), query=query))
            documents.append(Document(id=str(document_id), content=doc))
            qrels.append(QRel(query_id=str(query_id), document_id=str(document_id), score=1))
            document_id += 1

        return queries, documents, qrels

if __name__ == "__main__":
    ingestor = QMSumIngestor()
    queries, documents, qrels = ingestor.ingest()
    print(len(queries))
    print(len(documents))
    print(len(qrels))