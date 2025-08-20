from typing import override
from pathlib import Path
from evals.common import Document, QRel, Query
from evals.ingestors.common import BaseIngestor
from evals.utils import ROOT


class AirbnbIngestor(BaseIngestor):
    @override
    def dataset_id(self) -> str:
        return "evals/airbnb";
    
    @override  
    def ingest(self) -> tuple[list[Query], list[Document], list[QRel]]:
        dataset_path = Path(ROOT) / "data/datasets/airbnb"
        documents: list[Document] = []
        queries: list[Query] = []
        qrels: list[QRel] = []

        with open(dataset_path / "queries.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                query = Query.model_validate_json(line)
                queries.append(query)

        with open(dataset_path / "documents.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                document = Document.model_validate_json(line)
                documents.append(document)

        with open(dataset_path / "qrels.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                qrel = QRel.model_validate_json(line)
                qrels.append(qrel)

        return queries, documents, qrels