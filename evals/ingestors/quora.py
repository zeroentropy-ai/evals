from typing import Any, cast, override

import numpy as np
from datasets import (  # pyright: ignore[reportMissingTypeStubs]
    load_dataset,  # pyright: ignore[reportUnknownVariableType]
)
from tqdm.asyncio import tqdm

from evals.common import Document, QRel, Query
from evals.ingestors.common import BaseIngestor, clean_dataset


class QuoraIngestor(BaseIngestor):
    @override
    def dataset_id(self) -> str:
        return "evals/quora"
    
    @override
    def ingest(self) -> tuple[list[Query], list[Document], list[QRel]]:
        dataset_name = "BeIR/quora-generated-queries"

        overall_dataset = cast(Any, load_dataset(dataset_name))["train"]
        
        queries: list[Query] = []
        documents: list[Document] = []
        qrels: list[QRel] = []

        total_length = len(overall_dataset)
        random_indices = np.random.choice(range(total_length), 10000, replace=False)

        for i in tqdm(random_indices, desc="Datapoints"):
            queries.append(Query(id=str(i), query=overall_dataset[int(i)]["query"]))
            documents.append(Document(id=str(i), content=overall_dataset[int(i)]["text"]))
            qrels.append(QRel(query_id=str(i), document_id=str(i), score=1))

        return clean_dataset(queries, documents, qrels)