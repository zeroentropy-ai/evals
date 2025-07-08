from typing import Any, override

import mteb  # pyright: ignore[reportMissingTypeStubs]

from evals.common import Document, QRel, Query
from evals.ingestors.common import BaseIngestor, clean_dataset

# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false


def extract_document_content(doc_data: str | dict[Any, Any] | Any) -> str:
    """Extract document content from different MTEB dataset formats"""
    if isinstance(doc_data, str):
        return doc_data
    elif isinstance(doc_data, dict):
        # Handle dict format like {'title': '...', 'text': '...'}
        title = doc_data.get("title", "")
        text = doc_data.get("text", "")
        # Combine title and text, handling empty titles
        if title and text:
            return f"{title}\n{text}"
        elif text:
            return text
        elif title:
            return title
        else:
            return ""
    else:
        # Fallback to string conversion
        return str(doc_data)


def extract_query_content(query_data: str | dict[Any, Any] | Any) -> str:
    """Extract query content from different MTEB dataset formats"""
    if isinstance(query_data, str):
        return query_data
    elif isinstance(query_data, dict):
        # For some datasets, queries come as {id: text} dict
        # Take the first value (query text)
        if len(query_data) == 1:
            return list(query_data.values())[0]
        # Or it might have a 'query' field
        elif "query" in query_data:
            return query_data["query"]
        else:
            # Fallback to the first string value found
            for value in query_data.values():
                if isinstance(value, str):
                    return value
            return str(query_data)
    else:
        # Fallback to string conversion
        return str(query_data)


def extract_score_value(score_data: int | float | str | dict[Any, Any] | Any) -> float:
    """Extract score value from different MTEB dataset formats"""
    if isinstance(score_data, int | float):
        return float(score_data)
    elif isinstance(score_data, str):
        try:
            return float(score_data)
        except ValueError:
            # If can't convert string to float, return 1.0 as default
            return 1.0
    elif isinstance(score_data, dict):
        # Some datasets might have scores as dicts with 'score' field or similar
        if "score" in score_data:
            return float(score_data["score"])
        elif "relevance" in score_data:
            return float(score_data["relevance"])
        else:
            # Try to find the first numeric value
            for value in score_data.values():
                try:
                    return float(value)
                except (ValueError, TypeError):
                    continue
            # If no numeric value found, return 1.0 as default
            return 1.0
    else:
        # Fallback - assume binary relevance
        return 1.0


def find_best_split(
    available_splits: list[str], requested_split: str, language: str | None
) -> str:
    """Find the best available split based on requested split and language"""
    # First try exact match
    if requested_split in available_splits:
        return requested_split

    # For language-pair datasets (like BelebeleRetrieval), try to find language-specific splits
    if language and "-" in available_splits[0]:  # Language-pair format
        # Convert language codes like 'ara-Arab' to the format used in splits
        lang_variants = [language, language.replace("-", "_")]

        # Look for splits containing the language
        for split in available_splits:
            for variant in lang_variants:
                if variant in split:
                    return split

        # Look for Arabic variants if language contains 'ara'
        if "ara" in language.lower() or "arab" in language.lower():
            for split in available_splits:
                if "arab" in split.lower() or "ara" in split.lower():
                    return split

    # Fallback to first available split
    return available_splits[0]


class MasterMtebIngestor(BaseIngestor):
    task_name: str
    dataset_name: str
    language: str | None
    split: str

    def __init__(
        self,
        task_name: str,
        *,
        dataset_name: str,
        language: str | None = None,
        split: str = "test",
    ) -> None:
        self.task_name = task_name
        self.dataset_name = dataset_name
        self.language = language
        self.split = split

    @override
    def dataset_id(self) -> str:
        return f"evals/mteb/{self.task_name.lower()}"

    @override
    def ingest(self) -> tuple[list[Query], list[Document], list[QRel]]:
        """Load MTEB dataset and return queries, documents, qrels"""
        # Load MTEB task and data
        task: Any = mteb.get_task(self.task_name)
        task.load_data()

        # If language is multilingual, combine all splits
        if self.language == "multilingual":
            available_splits = (
                list(task.queries.keys())
                if hasattr(task, "queries") and task.queries
                else []
            )
            print(
                f"Language is multilingual, combining all {len(available_splits)} splits"
            )
            return self._combine_all_splits(task, available_splits)

        # Check if the requested split exists
        try:
            queries_dict = task.queries[self.split]
            corpus_dict = task.corpus[self.split]
            relevant_docs_dict = task.relevant_docs[self.split]
        except KeyError:
            available_splits = (
                list(task.queries.keys())
                if hasattr(task, "queries") and task.queries
                else []
            )
            print(f"WARNING: Split '{self.split}' not found for {self.task_name}")
            print(f"Available splits: {available_splits}")
            # Use smart split selection
            if available_splits:
                actual_split = find_best_split(
                    available_splits, self.split, self.language
                )
                print(f"Using split '{actual_split}' instead")
                queries_dict = task.queries[actual_split]
                corpus_dict = task.corpus[actual_split]
                relevant_docs_dict = task.relevant_docs[actual_split]
            else:
                raise ValueError(
                    f"No data splits available for {self.task_name}"
                ) from None

        # Handle nested structure (some datasets have queries/docs nested under split names)
        # Check if we have a nested structure like {'dev': {'Q1': 'text'}, 'test': {'Q2': 'text'}}

        # Check queries for nested structure
        if queries_dict and len(queries_dict) > 0:
            first_key = list(queries_dict.keys())[0]
            first_value = queries_dict[first_key]
            # If the value is a dict and the dict contains OTHER dicts (indicating nested splits), it's nested
            if isinstance(first_value, dict) and len(first_value) > 0:
                # Check if the inner dict has values that are strings/dicts (query content)
                # Normal MTEB query format: {'query_id': 'query text'} or {'query_id': {'query': 'text'}}
                # Nested format would be: {'split1': {'query1': 'text'}, 'split2': {...}}
                # We detect nested by checking if the inner dict contains query-like keys
                inner_sample = list(first_value.values())[0]
                # Only consider nested if the inner dict contains multiple query-like entries
                if isinstance(inner_sample, str | dict) and len(first_value) > 1:
                    print(f"Detected nested query structure for {self.task_name}")
                    flat_queries_dict = {}
                    for _inner_split, inner_queries in queries_dict.items():
                        flat_queries_dict.update(inner_queries)
                    queries_dict = flat_queries_dict

        # Check corpus for nested structure
        if corpus_dict and len(corpus_dict) > 0:
            first_key = list(corpus_dict.keys())[0]
            first_value = corpus_dict[first_key]
            # If the value is a dict and the dict contains OTHER dicts (not just text/title), it's nested
            if isinstance(first_value, dict) and len(first_value) > 0:
                # Check if the inner dict has values that are dicts (indicating nested splits)
                # Normal MTEB format is {'text': 'content'} or {'title': 'x', 'text': 'y'}
                # Nested format would be {'split1': {'doc1': {'text': 'content'}}, 'split2': {...}}
                inner_sample = list(first_value.values())[0]
                if isinstance(inner_sample, dict) and not (
                    "text" in first_value or "title" in first_value
                ):
                    print(f"Detected nested corpus structure for {self.task_name}")
                    flat_corpus_dict = {}
                    for _inner_split, inner_corpus in corpus_dict.items():
                        flat_corpus_dict.update(inner_corpus)
                    corpus_dict = flat_corpus_dict

        # Check qrels for nested structure
        if relevant_docs_dict and len(relevant_docs_dict) > 0:
            first_key = list(relevant_docs_dict.keys())[0]
            first_value = relevant_docs_dict[first_key]
            # If the value is a dict and contains dicts as values (query -> {doc_id: score}), it's nested
            if isinstance(first_value, dict) and len(first_value) > 0:
                # Check if the inner dict has values that are dicts (doc_id -> score mappings)
                inner_sample = list(first_value.values())[0]
                if isinstance(inner_sample, dict):  # Should be {doc_id: score} mapping
                    print(f"Detected nested qrels structure for {self.task_name}")
                    flat_relevant_docs_dict = {}
                    for _inner_split, inner_qrels in relevant_docs_dict.items():
                        for query_id, doc_scores in inner_qrels.items():
                            flat_relevant_docs_dict[query_id] = doc_scores
                    relevant_docs_dict = flat_relevant_docs_dict

        # Create Query objects
        queries = []
        for query_id, query_data in queries_dict.items():
            queries.append(
                Query(
                    id=query_id,
                    query=extract_query_content(query_data),
                    metadata={"dataset": self.dataset_name, "language": self.language},
                )
            )

        # Create Document objects
        documents = []
        for doc_id, doc_data in corpus_dict.items():
            documents.append(
                Document(
                    id=doc_id,
                    content=extract_document_content(doc_data),
                    metadata={"dataset": self.dataset_name, "language": self.language},
                )
            )

        # Create QRel objects
        qrels = []
        for query_id, doc_scores in relevant_docs_dict.items():
            for doc_id, score in doc_scores.items():
                qrels.append(
                    QRel(
                        query_id=query_id,
                        document_id=doc_id,
                        score=extract_score_value(score),
                    )
                )

        return clean_dataset(queries, documents, qrels)

    def _combine_all_splits(
        self, task: Any, available_splits: list[str]
    ) -> tuple[list[Query], list[Document], list[QRel]]:
        """Combine all splits into one multilingual dataset"""
        all_queries = []
        all_documents = []

        # Keep track of seen document IDs to avoid duplicates
        seen_doc_ids = set()
        # Keep track of seen query IDs to avoid duplicates
        seen_query_ids = set()

        # Step 1: Collect all documents from all splits
        for split in available_splits:
            try:
                corpus_dict = task.corpus[split]

                # Handle nested structure (some datasets have docs nested under split names)
                if corpus_dict and isinstance(list(corpus_dict.values())[0], dict):
                    # Nested structure: flatten all documents from all inner splits
                    for inner_split, inner_corpus in corpus_dict.items():
                        for doc_id, doc_data in inner_corpus.items():
                            if doc_id not in seen_doc_ids:
                                seen_doc_ids.add(doc_id)
                                all_documents.append(
                                    Document(
                                        id=doc_id,
                                        content=extract_document_content(doc_data),
                                        metadata={
                                            "dataset": self.dataset_name,
                                            "language": "multilingual",
                                            "split": f"{split}_{inner_split}",
                                        },
                                    )
                                )
                else:
                    # Direct structure
                    for doc_id, doc_data in corpus_dict.items():
                        if doc_id not in seen_doc_ids:
                            seen_doc_ids.add(doc_id)
                            all_documents.append(
                                Document(
                                    id=doc_id,
                                    content=extract_document_content(doc_data),
                                    metadata={
                                        "dataset": self.dataset_name,
                                        "language": "multilingual",
                                        "split": split,
                                    },
                                )
                            )

            except KeyError as e:
                print(f"Warning: Could not load corpus for split '{split}': {e}")
                continue

        # Step 2: Collect all queries from all splits
        for split in available_splits:
            try:
                queries_dict = task.queries[split]

                # Handle nested structure (some datasets have queries nested under split names)
                if queries_dict and isinstance(list(queries_dict.values())[0], dict):
                    # Nested structure: flatten all queries from all inner splits
                    for inner_split, inner_queries in queries_dict.items():
                        for query_id, query_data in inner_queries.items():
                            if query_id not in seen_query_ids:
                                seen_query_ids.add(query_id)
                                all_queries.append(
                                    Query(
                                        id=query_id,
                                        query=extract_query_content(query_data),
                                        metadata={
                                            "dataset": self.dataset_name,
                                            "language": "multilingual",
                                            "split": f"{split}_{inner_split}",
                                        },
                                    )
                                )
                else:
                    # Direct structure
                    for query_id, query_data in queries_dict.items():
                        if query_id not in seen_query_ids:
                            seen_query_ids.add(query_id)
                            all_queries.append(
                                Query(
                                    id=query_id,
                                    query=extract_query_content(query_data),
                                    metadata={
                                        "dataset": self.dataset_name,
                                        "language": "multilingual",
                                        "split": split,
                                    },
                                )
                            )

            except KeyError as e:
                print(f"Warning: Could not load queries for split '{split}': {e}")
                continue

        # Step 3: Create one cycle of qrels based on what actually exists
        all_qrels = []
        collected_doc_ids = seen_doc_ids
        collected_query_ids = seen_query_ids

        for split in available_splits:
            try:
                relevant_docs_dict = task.relevant_docs[split]

                # Handle nested structure (some datasets have qrels nested under split names)
                if relevant_docs_dict and isinstance(
                    list(relevant_docs_dict.values())[0], dict
                ):
                    # Check if this is nested qrels structure
                    first_value = list(relevant_docs_dict.values())[0]
                    # If the first value is a dict but doesn't contain document scores, it's nested
                    if isinstance(first_value, dict) and not any(
                        isinstance(v, int | float | str) for v in first_value.values()
                    ):
                        # Nested structure: process all inner qrels
                        for inner_qrels in relevant_docs_dict.values():
                            for query_id, doc_scores in inner_qrels.items():
                                if (
                                    query_id in collected_query_ids
                                ):  # Only if we have this query
                                    for doc_id, score in doc_scores.items():
                                        if (
                                            doc_id in collected_doc_ids
                                        ):  # Only if we have this document
                                            all_qrels.append(
                                                QRel(
                                                    query_id=query_id,
                                                    document_id=doc_id,
                                                    score=extract_score_value(score),
                                                )
                                            )
                    else:
                        # Direct qrels structure
                        for query_id, doc_scores in relevant_docs_dict.items():
                            if (
                                query_id in collected_query_ids
                            ):  # Only if we have this query
                                for doc_id, score in doc_scores.items():
                                    if (
                                        doc_id in collected_doc_ids
                                    ):  # Only if we have this document
                                        all_qrels.append(
                                            QRel(
                                                query_id=query_id,
                                                document_id=doc_id,
                                                score=extract_score_value(score),
                                            )
                                        )
                else:
                    # Direct structure
                    for query_id, doc_scores in relevant_docs_dict.items():
                        if (
                            query_id in collected_query_ids
                        ):  # Only if we have this query
                            for doc_id, score in doc_scores.items():
                                if (
                                    doc_id in collected_doc_ids
                                ):  # Only if we have this document
                                    all_qrels.append(
                                        QRel(
                                            query_id=query_id,
                                            document_id=doc_id,
                                            score=extract_score_value(score),
                                        )
                                    )

            except KeyError as e:
                print(f"Warning: Could not load qrels for split '{split}': {e}")
                continue

        print(
            f"Combined: {len(all_queries)} queries, {len(all_documents)} documents, {len(all_qrels)} qrels"
        )

        # Additional safety check: filter out qrels that reference non-existent queries/documents
        # This can happen with complex multilingual datasets where splits have different ID spaces
        valid_qrels = []
        seen_qrel_pairs = (
            set()
        )  # Track (query_id, document_id) pairs to avoid duplicates

        for qrel in all_qrels:
            if (
                qrel.query_id in collected_query_ids
                and qrel.document_id in collected_doc_ids
            ):
                qrel_pair = (qrel.query_id, qrel.document_id)
                if qrel_pair not in seen_qrel_pairs:
                    seen_qrel_pairs.add(qrel_pair)
                    valid_qrels.append(qrel)

        if len(valid_qrels) != len(all_qrels):
            print(
                f"Filtered qrels: {len(all_qrels)} -> {len(valid_qrels)} (removed {len(all_qrels) - len(valid_qrels)} duplicates/invalid references)"
            )

        return clean_dataset(all_queries, all_documents, valid_qrels)

    async def load_queries(self) -> list[Query]:
        # Load MTEB task and data
        task: Any = mteb.get_task(self.task_name)
        task.load_data()

        # Extract queries from the task attributes
        try:
            queries_dict = task.queries[self.split]
        except KeyError:
            available_splits = (
                list(task.queries.keys())
                if hasattr(task, "queries") and task.queries
                else []
            )
            print(f"WARNING: Split '{self.split}' not found for {self.task_name}")
            print(f"Available splits: {available_splits}")
            if available_splits:
                actual_split = find_best_split(
                    available_splits, self.split, self.language
                )
                print(f"Using split '{actual_split}' instead")
                queries_dict = task.queries[actual_split]
            else:
                raise ValueError(
                    f"No query splits available for {self.task_name}"
                ) from None

        queries = []
        for query_id, query_data in queries_dict.items():
            queries.append(
                Query(
                    id=query_id,
                    query=extract_query_content(query_data),
                    metadata={"dataset": self.dataset_name, "language": self.language},
                )
            )

        return queries

    async def load_corpus(self) -> list[Document]:
        # Load MTEB task and data
        task: Any = mteb.get_task(self.task_name)
        task.load_data()

        # Extract corpus from the task attributes
        try:
            corpus_dict = task.corpus[self.split]
        except KeyError:
            available_splits = (
                list(task.corpus.keys())
                if hasattr(task, "corpus") and task.corpus
                else []
            )
            print(f"WARNING: Split '{self.split}' not found for {self.task_name}")
            print(f"Available splits: {available_splits}")
            if available_splits:
                actual_split = find_best_split(
                    available_splits, self.split, self.language
                )
                print(f"Using split '{actual_split}' instead")
                corpus_dict = task.corpus[actual_split]
            else:
                raise ValueError(
                    f"No corpus splits available for {self.task_name}"
                ) from None

        documents = []
        for doc_id, doc_data in corpus_dict.items():
            documents.append(
                Document(
                    id=doc_id,
                    content=extract_document_content(doc_data),
                    metadata={"dataset": self.dataset_name, "language": self.language},
                )
            )

        return documents

    async def load_qrels(self) -> list[QRel]:
        # Load MTEB task and data
        task: Any = mteb.get_task(self.task_name)
        task.load_data()

        # Extract relevance judgments from the task attributes
        try:
            relevant_docs_dict = task.relevant_docs[self.split]
        except KeyError:
            available_splits = (
                list(task.relevant_docs.keys())
                if hasattr(task, "relevant_docs") and task.relevant_docs
                else []
            )
            print(f"WARNING: Split '{self.split}' not found for {self.task_name}")
            print(f"Available splits: {available_splits}")
            if available_splits:
                actual_split = find_best_split(
                    available_splits, self.split, self.language
                )
                print(f"Using split '{actual_split}' instead")
                relevant_docs_dict = task.relevant_docs[actual_split]
            else:
                raise ValueError(
                    f"No qrel splits available for {self.task_name}"
                ) from None

        qrels = []
        for query_id, doc_scores in relevant_docs_dict.items():
            for doc_id, score in doc_scores.items():
                qrels.append(
                    QRel(
                        query_id=query_id,
                        document_id=doc_id,
                        score=extract_score_value(score),
                    )
                )

        return qrels

    def load_dataset(self) -> dict[str, Any]:
        """Load the raw MTEB dataset for inspection"""
        task: Any = mteb.get_task(self.task_name)
        task.load_data()

        return {
            "corpus": task.corpus,
            "queries": task.queries,
            "relevant_docs": task.relevant_docs,
            "metadata": {
                "task_name": self.task_name,
                "dataset_name": self.dataset_name,
                "task_type": task.metadata.type,
                "eval_splits": task.metadata.eval_splits,
                "eval_langs": task.metadata.eval_langs,
                "license": task.metadata.license,
            },
        }
