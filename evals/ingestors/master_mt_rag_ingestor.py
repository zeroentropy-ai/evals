import csv
import io
import json
import zipfile
from typing import override

import httpx
from tqdm import tqdm

from evals.common import Document, QRel, Query
from evals.ingestors.common import BaseIngestor, clean_dataset


class MasterMtRagIngestor(BaseIngestor):
    dataset_name: str

    def __init__(self, dataset_name: str) -> None:
        self.dataset_name = dataset_name

    @override
    def dataset_id(self) -> str:
        return f"evals/mt_rag/{self.dataset_name}"

    @override
    def ingest(self) -> tuple[list[Query], list[Document], list[QRel]]:
        # Download and parse corpus
        corpus_url = f"https://raw.githubusercontent.com/IBM/mt-rag-benchmark/refs/heads/main/corpora/{self.dataset_name}.jsonl.zip"
        documents = self._download_and_parse_corpus(corpus_url)

        # Download and parse questions
        questions_url = f"https://raw.githubusercontent.com/IBM/mt-rag-benchmark/refs/heads/main/human/retrieval_tasks/{self.dataset_name}/{self.dataset_name}_questions.jsonl"
        queries = self._download_and_parse_questions(questions_url)

        # Download and parse qrels
        qrels_url = f"https://raw.githubusercontent.com/IBM/mt-rag-benchmark/refs/heads/main/human/retrieval_tasks/{self.dataset_name}/qrels/dev.tsv"
        qrels = self._download_and_parse_qrels(qrels_url)

        return clean_dataset(queries, documents, qrels)

    def _download_and_parse_corpus(self, url: str) -> list[Document]:
        """Download and parse the corpus JSONL file from a ZIP archive."""
        print(f"Downloading corpus from {url}")

        response = httpx.get(url)
        response.raise_for_status()

        # Extract JSONL from ZIP in memory
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
            # Get the first (and presumably only) file in the ZIP
            jsonl_filename = zip_file.namelist()[0]
            with zip_file.open(jsonl_filename) as jsonl_file:
                jsonl_content = jsonl_file.read().decode("utf-8")

        documents: list[Document] = []
        for line in tqdm(jsonl_content.strip().split("\n"), desc="Parsing corpus"):
            if len(line.strip()) == 0:
                continue
            data = json.loads(line)
            # Format: {"_id": ..., "text": ..., "title": ..., "metadata": ...}
            if "_id" in data:
                document_id = str(data["_id"])
            else:
                # Sometimes this happens
                document_id = str(data["document_id"])
            content = str(data["text"])
            metadata = data.get("metadata", {})
            if (
                "title" in data
                and isinstance(data["title"], str)
                and len(data["title"].strip()) > 0
            ):
                assert "title" not in metadata
                metadata["title"] = data["title"].strip()

            documents.append(
                Document(
                    id=document_id,
                    content=content,
                    metadata=metadata,
                )
            )

        return documents

    def _download_and_parse_questions(self, url: str) -> list[Query]:
        """Download and parse the questions JSONL file."""
        print(f"Downloading questions from {url}")

        response = httpx.get(url)
        response.raise_for_status()

        queries: list[Query] = []
        for line in tqdm(response.text.strip().split("\n"), desc="Parsing questions"):
            if len(line.strip()) == 0:
                continue
            data = json.loads(line)
            # Format: {"_id": ..., "text": ...}
            queries.append(
                Query(
                    id=str(data["_id"]),
                    query=str(data["text"]),
                    metadata={},
                )
            )

        return queries

    def _download_and_parse_qrels(self, url: str) -> list[QRel]:
        """Download and parse the qrels TSV file."""
        print(f"Downloading qrels from {url}")

        response = httpx.get(url)
        response.raise_for_status()

        qrels: list[QRel] = []
        reader = csv.DictReader(io.StringIO(response.text), delimiter="\t")

        qrel_pkey_to_score: dict[tuple[str, str], int] = {}

        for row in tqdm(reader, desc="Parsing qrels"):
            query_id = str(row["query-id"])
            # Ignore the final "-0-chunk_offset"
            document_id = str(row["corpus-id"]).rsplit("-", 2)[0]
            score = int(row["score"])

            # Remove duplicates with the same score
            pkey = (query_id, document_id)
            if pkey in qrel_pkey_to_score:
                assert qrel_pkey_to_score[pkey] == score
                continue
            qrel_pkey_to_score[pkey] = score

            qrels.append(
                QRel(
                    query_id=query_id,
                    document_id=document_id,
                    score=score,
                )
            )

        return qrels
