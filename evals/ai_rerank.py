from typing import Literal

import diskcache as dc  # pyright: ignore[reportMissingTypeStubs]
from pydantic import BaseModel

from evals.ai import AIMessage, AIModel, ai_call
from evals.utils import clamp


class AIModelAsReranker(BaseModel):
    model: AIModel
    rerank_type: Literal["listwise"]


class RerankResult(BaseModel):
    index: int
    score: float


class RerankOutput(BaseModel):
    results: list[RerankResult]


async def ai_rerank_by_ai_model(
    model: AIModelAsReranker,
    query: str,
    documents: list[str],
    *,
    max_bytes: int,
    # Cache
    cache: dc.Cache | None = None,
) -> list[float]:
    remaining_bytes = max_bytes
    truncated_documents: list[str] = []
    for document in documents:
        if remaining_bytes < 100:
            break
        truncated_document = (
            document.strip().encode()[:remaining_bytes].decode(errors="ignore").strip()
        )
        truncated_documents.append(truncated_document)
        remaining_bytes -= len(truncated_document.encode())

    documents_string = "# DOCUMENTS\n\n"
    for i, truncated_document in enumerate(truncated_documents):
        documents_string += f"## Document Index={i}\n"
        documents_string += f"{truncated_document}\n\n"

    output = await ai_call(
        model.model,
        messages=[
            AIMessage(
                role="system",
                content=f"""
You are a reranker. You will be given a query, and a list of documents, and your task is to score each document for how relevant it is to the query. The score should be between 0.0 and 1.0.

# QUERY

{query}
""",
            ),
            AIMessage(
                role="user",
                content=documents_string,
            ),
        ],
        response_format=RerankOutput,
        cache=cache,
    )

    scores = [0.0 for _ in range(len(documents))]
    for result in output.results:
        if result.index < 0 or result.index >= len(scores):
            continue
        scores[result.index] = clamp(result.score, 0, 1)

    return scores
