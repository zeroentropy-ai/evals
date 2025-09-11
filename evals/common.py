from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, Field, computed_field

from evals.utils import ROOT

RetrievalMethod = Literal[
    "qwen3_4b", "qwen3_0.6b", "voyageai", "openai_small", "bm25", "hybrid"
]
MergeStatus = Literal["merged", "unmerged"]
RerankerName = Literal[
    "cohere",
    "salesforce",
    "zeroentropy-large",
    "zeroentropy-small",
    "zeroentropy-small-modal",
    "zeroentropy-large-modal",
    "zeroentropy-baseten",
    "mixedbread",
    "jina",
    "qwen",
    "openai-large-embedding",
    "gpt-4o-mini",
    "gpt-4.1-mini",
    "gpt-5-mini",
    "gpt-5-nano",
]


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

    def ze_results_path(
        self, retrieval_method: RetrievalMethod, include_relevant_docs: bool
    ) -> str:
        merge_status: MergeStatus = "merged" if include_relevant_docs else "unmerged"
        return self.file_path(f"{retrieval_method}/{merge_status}/ze_results.jsonl")

    def embeddings_cache_path(
        self, retrieval_method: RetrievalMethod, include_relevant_docs: bool
    ) -> str:
        merge_status: MergeStatus = "merged" if include_relevant_docs else "unmerged"
        return self.file_path(f"{retrieval_method}/{merge_status}/embeddings_cache.db")

    def latest_ze_results_path(
        self,
        retrieval_method: RetrievalMethod,
        include_relevant_docs: bool,
        reranker: RerankerName,
    ) -> str:
        merge_status: MergeStatus = "merged" if include_relevant_docs else "unmerged"
        return self.file_path(
            f"{retrieval_method}/{merge_status}/{reranker}/latest_ze_results.jsonl"
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
