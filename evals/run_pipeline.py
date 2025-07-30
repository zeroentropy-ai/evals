import asyncio
from typing import Literal

from evals.common import RerankerName, RetrievalMethod
from evals.ingestors.common import BaseIngestor
from evals.run_embeddings import run_embeddings
from evals.run_ingestors import run_ingestors
from evals.run_ndcg import run_ndcg
from evals.run_rerankers import run_rerankers
from evals.types import OLD_INGESTORS

INGESTORS: list[BaseIngestor] = OLD_INGESTORS
RETRIEVAL_METHOD: RetrievalMethod = "openai_small"
INCLUDE_RELEVANT_DOCS: bool = True
RERANKERS: list[RerankerName] = ["zeroentropy-large"]
MAX_QUERIES = 100

Action = Literal["ingestors", "embeddings", "rerankers", "ndcg"]
ORDER: dict[Action, int] = {
    "ingestors": 0,
    "embeddings": 1,
    "rerankers": 2,
    "ndcg": 3,
}


async def run_pipeline(
    start_action: Action = "ingestors", end_action: Action = "ndcg"
) -> None:
    # Run actions, but only between the requested start and end inclusive
    if (
        ORDER[start_action] <= ORDER["ingestors"]
        and ORDER[end_action] >= ORDER["ingestors"]
    ):
        run_ingestors(ingestors=INGESTORS, max_queries=MAX_QUERIES)
    if (
        ORDER[start_action] <= ORDER["embeddings"]
        and ORDER[end_action] >= ORDER["embeddings"]
    ):
        await run_embeddings(
            ingestors=INGESTORS,
            retrieval_method=RETRIEVAL_METHOD,
            include_relevant_docs=INCLUDE_RELEVANT_DOCS,
        )
    if (
        ORDER[start_action] <= ORDER["rerankers"]
        and ORDER[end_action] >= ORDER["rerankers"]
    ):
        await run_rerankers(
            ingestors=INGESTORS,
            rerankers=RERANKERS,
            retrieval_method=RETRIEVAL_METHOD,
            include_relevant_docs=INCLUDE_RELEVANT_DOCS,
        )
    if ORDER[start_action] <= ORDER["ndcg"] and ORDER[end_action] >= ORDER["ndcg"]:
        run_ndcg(
            ingestors=INGESTORS,
            retrieval_method=RETRIEVAL_METHOD,
            include_relevant_docs=INCLUDE_RELEVANT_DOCS,
            rerankers=RERANKERS,
        )


if __name__ == "__main__":
    asyncio.run(run_pipeline("ingestors", "ndcg"))
