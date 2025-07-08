from typing import Any, cast, override

from datasets import (  # pyright: ignore[reportMissingTypeStubs]
    load_dataset,  # pyright: ignore[reportUnknownVariableType]
)
from tqdm.asyncio import tqdm

from evals.common import Document, QRel, Query
from evals.ingestors.common import BaseIngestor, clean_dataset


class FinancebenchIngestor(BaseIngestor):
    @override
    def dataset_id(self) -> str:
        return "evals/financebench"

    @override
    def ingest(self) -> tuple[list[Query], list[Document], list[QRel]]:
        dataset_name = "PatronusAI/financebench"
        dataset = cast(Any, load_dataset(path=dataset_name))
        dataset = dataset["train"]

        # Create queries
        queries: list[Query] = []
        for i, question in tqdm(enumerate(dataset["question"]), desc="Queries"):
            queries.append(Query(id=str(i), query=question, metadata={}))

        # Create documents
        documents: list[Document] = []
        document_id = 0
        for evidence_list in tqdm(dataset["evidence"], desc="Documents"):
            if isinstance(evidence_list, list):
                evidence_list = cast(list[Any], evidence_list)
                for evidence in evidence_list:
                    documents.append(
                        Document(
                            id=str(document_id),
                            content=evidence["evidence_text"],
                        )
                    )
                    document_id += 1
            else:
                documents.append(
                    Document(id=str(document_id), content=evidence_list, metadata={})
                )
                document_id += 1

        # Create qrels
        qrels: list[QRel] = []
        document_id = 0
        for i, evidence_list in tqdm(enumerate(dataset["evidence"]), desc="Documents"):
            if isinstance(evidence_list, list):
                evidence_list = cast(list[Any], evidence_list)
                for _ in evidence_list:
                    qrels.append(
                        QRel(query_id=str(i), document_id=str(document_id), score=1.0)
                    )
                    document_id += 1
            else:
                qrels.append(
                    QRel(query_id=str(i), document_id=str(document_id), score=1.0)
                )
                document_id += 1

        return clean_dataset(queries, documents, qrels)
