from typing import Any

from pydantic import AliasChoices, BaseModel, Field, computed_field

from evals.utils import ROOT


class ZEDataset(BaseModel):
    id: str

    @classmethod
    def from_list(cls, dataset_ids: list[str]) -> list["ZEDataset"]:
        return [cls(id=dataset_id) for dataset_id in dataset_ids]

    @computed_field
    @property
    def root_path(self) -> str:
        return f"{ROOT}/data/datasets/{self.id}"

    def file_path(self, relative_file_path: str) -> str:
        return f"{self.root_path}/{relative_file_path}"

    @computed_field
    @property
    def queries_path(self) -> str:
        return self.file_path("queries.jsonl")

    @computed_field
    @property
    def documents_path(self) -> str:
        return self.file_path("documents.jsonl")

    @computed_field
    @property
    def qrels_path(self) -> str:
        return self.file_path("qrels.jsonl")

    def retrieval_method_path(
        self,
        retrieval_method: str,
        include_relevant_docs: bool,
        path: str,
    ) -> str:
        if include_relevant_docs:
            retrieval_method = f"{retrieval_method}+include_relevant_docs"
        return self.file_path(f"{retrieval_method}/{path}")

    def ze_results_path(
        self,
        retrieval_method: str,
        include_relevant_docs: bool,
    ) -> str:
        return self.retrieval_method_path(
            retrieval_method, include_relevant_docs, "ze_results.jsonl"
        )

    def embeddings_cache_path(
        self,
        retrieval_method: str,
    ) -> str:
        return self.file_path(f"embeddings_cache/{retrieval_method}.db")

    def reranker_cache_path(
        self,
        reranker: str,
    ) -> str:
        return self.file_path(f"reranker_cache/{reranker}.db")

    def ze_scores_path(
        self,
        retrieval_method: str,
        include_relevant_docs: bool,
        reranker: str,
    ) -> str:
        return self.retrieval_method_path(
            retrieval_method, include_relevant_docs, f"{reranker}/ze_scores.jsonl"
        )


class Document(BaseModel):
    id: str = Field(validation_alias=AliasChoices("document_id", "id"))
    content: str
    metadata: dict[str, Any] = {}
    scores: dict[str, float] = {}

    def format_string(self) -> str:
        title = self.metadata.get("title", None)
        if isinstance(title, str) and len(title) > 0:
            return f"Title: {title}\nContent: {self.content}"
        else:
            return self.content


class Query(BaseModel):
    id: str
    query: str
    metadata: dict[str, Any] = {}


class QRel(BaseModel):
    query_id: str
    document_id: str
    score: float


# ze_results.jsonl, where each line is this object
class ZEResults(BaseModel):
    query_id: str
    query: str
    documents: list[Document]


class DocumentScores(BaseModel):
    document_id: str
    scores: dict[str, float]


class QueryScores(BaseModel):
    query_id: str
    documents: list[DocumentScores]
