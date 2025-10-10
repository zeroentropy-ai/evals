from typing import Any, cast, override

from datasets import (  # pyright: ignore[reportMissingTypeStubs]
    concatenate_datasets,
    load_dataset,  # pyright: ignore[reportUnknownVariableType]
)
from tqdm.asyncio import tqdm

from evals.common import Document, QRel, Query
from evals.ingestors.common import BaseIngestor, clean_dataset


class MSMarcoIngestor(BaseIngestor):
    @override
    def dataset_id(self) -> str:
        return "evals/msmarco"

    @override
    def ingest(self) -> tuple[list[Query], list[Document], list[QRel]]:
        dataset_name = "microsoft/ms_marco"
        dataset = cast(Any, load_dataset(dataset_name, "v1.1"))
        test_data = concatenate_datasets([dataset["validation"], dataset["test"]])

        queries: list[Query] = []
        documents: list[Document] = []
        qrels: list[QRel] = []

        document_id = 0
        all_queries = test_data["query"]
        all_query_ids = test_data["query_id"]
        all_doc_info = test_data["passages"]
        for query, query_id, doc_info in tqdm(
            zip(all_queries, all_query_ids, all_doc_info, strict=True),
            desc="Datapoints",
        ):
            queries.append(Query(id=str(query_id), query=query))
            for doc, score in zip(
                doc_info["passage_text"], doc_info["is_selected"], strict=True
            ):
                documents.append(Document(id=str(document_id), content=doc))
                qrels.append(
                    QRel(
                        query_id=str(query_id),
                        document_id=str(document_id),
                        score=score,
                    )
                )
                document_id += 1

        return clean_dataset(queries, documents, qrels)


if __name__ == "__main__":
    ingestor = MSMarcoIngestor()
    queries, documents, qrels = ingestor.ingest()
    print(len(queries))
    print(len(documents))
    print(len(qrels))
