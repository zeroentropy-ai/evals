from typing import Any

from pydantic import AliasChoices, BaseModel, Field, computed_field

from utils import ROOT


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

    @computed_field
    @property
    def ze_results_path(self) -> str:
        return self.file_path("ze_results.jsonl")

    @computed_field
    @property
    def ai_scores_path(self) -> str:
        return self.file_path("ai_scores.json")


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


# Pairs Dataset


class Pair(BaseModel):
    dataset_id: str | None = None
    pair_id: str
    query_id: str
    query: str
    document_a: Document
    document_b: Document


class Pairs(BaseModel):
    pairs: list[Pair]


# Scored Pairs


class PairScore(BaseModel):
    thought: str
    score: float


class ScoredPair(BaseModel):
    pair: Pair
    openai_score: PairScore
    gemini_score: PairScore
    anthropic_score: PairScore


# ai_scores.json
class ScoredPairs(BaseModel):
    scored_pairs: list[ScoredPair]
