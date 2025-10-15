from typing import Any, cast, override

from datasets import (  # pyright: ignore[reportMissingTypeStubs]
    load_dataset,  # pyright: ignore[reportUnknownVariableType]
)
from tqdm import tqdm

from evals.common import Document, QRel, Query
from evals.ingestors.common import BaseIngestor, clean_dataset

class EnronQaIngestor(BaseIngestor):
    @override
    def dataset_id(self) -> str:
        return "evals/enronqa"

    @override
    def ingest(self) -> tuple[list[Query], list[Document], list[QRel]]:
        dataset_name = "corbt/enron-emails"
        queries_name = "corbt/enron_emails_sample_questions"
        # Load the datasets
        corpus_dataset = cast(Any, load_dataset(dataset_name, "default"))["train"]
        queries_dataset = cast(Any, load_dataset(queries_name, "default"))["train"]


        # Create documents
        documents: list[Document] = []
        message_id_to_doc_id: dict[str, int] = {}
        for index, document in enumerate(tqdm(corpus_dataset, desc="Documents")):
            doc_id = index
            email_content = \
                f"Subject: {document["subject"]}\n" + \
                f"Date: {document["date"].strftime("%Y-%m-%d %H:%M:%S")}\n\n" + \
                f"From: {document["from"]}\n" + \
                f"To: {', '.join(document["to"])}\n" + \
                f"Cc: {', '.join(document["cc"])}\n" + \
                f"Bcc: {', '.join(document["bcc"])}\n" + \
                f"{document["body"]}"

            message_id_to_doc_id[document["message_id"]] = doc_id
            documents.append(
                Document(
                    id=str(doc_id),
                    content=email_content,
                    metadata={}
                )
            )

        # Create QRel objects
        qrels: list[QRel] = []
        valid_query_ids: set[str] = set()
        for index, question in enumerate(tqdm(queries_dataset, "QRels")):
            related_message_ids = question["message_ids"]
            related_message_indices = [message_id_to_doc_id[mid] for mid in related_message_ids if mid in message_id_to_doc_id]
            if len(related_message_indices) > 0:
                for doc_index in related_message_indices:
                    qrels.append(
                        QRel(
                            query_id=str(index),
                            document_id=str(doc_index),
                            score=1.0,
                        )
                    )
                valid_query_ids.add(qrels[-1].query_id)

        # Create Query objects
        queries: list[Query] = []
        for index, question in enumerate(tqdm(queries_dataset, desc="Queries")):
            if str(index) not in valid_query_ids:
                continue
            queries.append(
                Query(
                    id=str(index),
                    query=question["question"],
                    metadata={},
                )
            )

        return clean_dataset(queries, documents, qrels)