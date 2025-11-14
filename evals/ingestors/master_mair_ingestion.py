import json
from random import Random
from typing import Any, override

import requests
from datasets import load_dataset  # pyright: ignore[reportMissingTypeStubs]

from evals.common import Document, QRel, Query
from evals.ingestors.common import (
    BaseIngestor,
    clean_dataset,
)

# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false


class MasterMairIngestor(BaseIngestor):
    task_name: str
    dataset_name: str
    split: str
    queries_split: str
    docs_split: str

    def __init__(
        self,
        task_name: str,
        queries_split: str,
        docs_split: str,
    ) -> None:
        self.task_name = task_name
        self.queries_split = queries_split
        self.docs_split = docs_split

        self.dataset_name = "".join(filter(str.isalnum, task_name)).lower()
        if queries_split != "queries":
            self.dataset_name += f"_{queries_split.removesuffix('queries').lower()}"

    @classmethod
    def all_splits(cls) -> list["MasterMairIngestor"]:
        queries_dataset_name = "MAIR-Bench/MAIR-Queries"
        docs_dataset_name = "MAIR-Bench/MAIR-Docs"

        def extract_subset_splits(dataset_name: str) -> dict[str, list[str]]:
            API_URL = (
                f"https://datasets-server.huggingface.co/splits?dataset={dataset_name}"
            )
            response = requests.get(API_URL)
            response.raise_for_status()
            data = response.json()

            # Extract (config, split) pairs from the response
            config_split_pairs = [
                (item["config"], item["split"]) for item in data["splits"]
            ]

            splits_by_subset: dict[str, list[str]] = {}
            for config, split in config_split_pairs:
                if config not in splits_by_subset:
                    splits_by_subset[config] = []
                splits_by_subset[config].append(split)
            return splits_by_subset

        query_subset_splits = extract_subset_splits(queries_dataset_name)
        doc_subset_splits = extract_subset_splits(docs_dataset_name)
        assert set(query_subset_splits.keys()) == set(doc_subset_splits.keys()), (
            "Query and Doc subsets do not match"
        )

        subset_splits = []
        for subset in query_subset_splits.keys():
            query_splits = set(
                split.removesuffix("queries") for split in query_subset_splits[subset]
            )
            doc_splits = set(
                split.removesuffix("docs") for split in doc_subset_splits[subset]
            )
            assert query_splits == doc_splits, (
                f"Query and Doc splits do not match for subset {subset}"
            )
            subset_splits.extend(
                [
                    (subset, query_split, doc_split)
                    for query_split, doc_split in zip(
                        sorted(query_subset_splits[subset]),
                        sorted(doc_subset_splits[subset]),
                        strict=True,
                    )
                ]
            )

        # Return an ingestor for each subset/split pair in MAIR-Bench
        ingestors: list[MasterMairIngestor] = []
        for subset, queries_split, docs_split in subset_splits:
            ingestors.append(
                MasterMairIngestor(
                    task_name=subset, queries_split=queries_split, docs_split=docs_split
                )
            )
        return ingestors

    @override
    def dataset_id(self) -> str:
        return f"evals/mair/{self.dataset_name}"

    @override
    def ingest(self) -> tuple[list[Query], list[Document], list[QRel]]:
        """Load MAIR task and return queries, documents, qrels"""

        # Load MAIR queries -- rows are {id, instruction, query, labels: {id, score}}
        queries_data = load_dataset(
            "MAIR-Bench/MAIR-Queries", self.task_name, split=self.queries_split
        )

        # Load MAIR documents -- rows are {id, doc}
        corpus_data = load_dataset(
            "MAIR-Bench/MAIR-Docs", self.task_name, split=self.docs_split
        )

        queries = self._load_queries(queries_data)
        documents = self._load_documents(corpus_data)
        qrels = self._load_qrels(queries_data)

        return clean_dataset(queries, documents, qrels)

    def _format_instruction_query(
        self, instruction: str, query: str, qid: str
    ) -> tuple[str, str]:
        # Format instruction/query pair to one of five single-string templates
        return Random(qid).choice(
            [
                ("newline", f"{instruction}\n\n{query}"),
                (
                    "xml",
                    f"<instruction>{instruction}</instruction>\n<query>{query}</query>",
                ),
                ("markdown", f"# Instruction\n{instruction}\n\n# Query\n{query}"),
                (
                    "json",
                    f'{{"instruction": "{json.dumps(instruction)}", "query": "{json.dumps(query)}"}}',
                ),
                ("caps", f"TASK DESCRIPTION: {instruction}\n\nINPUT: {query}"),
                ("xml2", f"<task>{instruction}</task>\n<input>{query}</input>"),
                (
                    "naturallanguage",
                    f"Instructions below:\n{instruction}\n\nNow please answer the following query:\n{query}",
                ),
            ]
        )

    def _load_queries(self, query_dataset: Any) -> list[Query]:
        queries: list[Query] = []
        seen_qids: set[str] = set()
        for idx, row in enumerate(query_dataset):
            # Handle potential duplicate query IDs
            query_id = str(row["qid"])
            if query_id in seen_qids:
                query_id = f"{query_id}_{idx}"
            seen_qids.add(query_id)

            template, query_text = self._format_instruction_query(
                instruction=row.get("instruction", ""),
                query=row["query"],
                qid=query_id,
            )
            queries.append(
                Query(
                    id=str(query_id),
                    query=query_text,
                    metadata={"dataset": self.dataset_name, "template": template},
                )
            )
        return queries

    def _load_documents(self, docs_dataset: Any) -> list[Document]:
        documents: list[Document] = []
        seen_doc_ids: set[str] = set()
        for row in docs_dataset:
            if str(row["id"]) in seen_doc_ids:
                continue
            seen_doc_ids.add(str(row["id"]))
            documents.append(
                Document(
                    id=str(row["id"]),
                    content=row["doc"],
                    metadata={"dataset": self.dataset_name},
                )
            )
        return documents

    def _load_qrels(self, query_dataset: Any) -> list[QRel]:
        qrels: list[QRel] = []
        qrel_pkey_to_score: dict[tuple[str, str], int] = {}
        seen_qids: set[str] = set()
        for idx, row in enumerate(query_dataset):
            # Handle potential duplicate query IDs
            query_id = str(row["qid"])
            if query_id in seen_qids:
                query_id = f"{query_id}_{idx}"
            seen_qids.add(query_id)

            for label in row["labels"]:
                doc_id = str(label["id"])
                score = int(label["score"])

                pkey = (query_id, doc_id)
                qrel_pkey_to_score[pkey] = max(qrel_pkey_to_score.get(pkey, 0), score)
        for (query_id, doc_id), score in qrel_pkey_to_score.items():
            qrels.append(
                QRel(
                    query_id=query_id,
                    document_id=doc_id,
                    score=score,
                )
            )
        return qrels
