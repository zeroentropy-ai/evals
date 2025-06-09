import asyncio
import time
from typing import Any, cast

from tqdm.asyncio import tqdm

from ml.training.pairwise.common import Batch, get_pairwise_dataset
from ml.utils import ROOT, flatten, format_globals
import aiohttp
from ml.test import TestModel, Input, Output

# pyright: reportUnknownMemberType=false

DATASET_PATH = f"{ROOT}/data/datasets/bioasq/ai_scores.json"

client: aiohttp.ClientSession | None = None


async def rerank_batch(batch: Batch) -> list[float]:
    datapoints = batch.datapoints

    query_documents: list[tuple[str, str]] = []
    for datapoint in datapoints:
        query_documents.append((datapoint.query, datapoint.document_a))
    for datapoint in datapoints:
        query_documents.append((datapoint.query, datapoint.document_b))

    url = "https://npip99--testing-ze-rerank-1-model-endpoint.modal.run"
    payload = {
        "query_documents": query_documents,
    }

    assert client is not None
    async with client.post(url, json=payload) as response:
        result = await response.json()
        scores = [float(score) for score in result["scores"]]

    score_diffs = [
        scores[len(datapoints) + i] - scores[i] for i in range(len(datapoints))
    ]
    assert len(score_diffs) == len(query_documents)
    return score_diffs


async def main() -> None:
    global client
    client = aiohttp.ClientSession()

    dataset = get_pairwise_dataset(
        DATASET_PATH,
        rank=0,
        size=1,
        batch_size_tokens=40_000,
        subset="test",
    )

    datapoints = flatten([batch.datapoints for batch in dataset])
    print(
        f"Total Characters: {sum(len(dp.query) + len(dp.document_a) + len(dp.document_b) for dp in datapoints)}"
    )
    t0 = time.time()
    batch_scores = await tqdm.gather(*[rerank_batch(batch) for batch in dataset])
    t1 = time.time()
    print(f"Total time: {t1 - t0:.3f}")
    # batch_scores = []
    # for batch in dataset:
    #     batch_scores.append(await rerank_batch(batch))
    scores = flatten(batch_scores)
    print(f"Total Pairs: {len(scores)}")

    # Calculate accuracy
    for only_consensus in [True, False]:
        num_correct = 0
        num_samples = 0
        for score, datapoint in zip(scores, datapoints, strict=True):
            if only_consensus and not datapoint.is_consensus:
                continue
            assert datapoint.ai_scores is not None
            target_score = sum(
                0.0 if x > 0 else (1.0 if x < 0 else 0.5) for x in datapoint.ai_scores
            ) / len(datapoint.ai_scores)
            pred_prefers_a = score > 0
            target_prefers_a = target_score < 0.5
            pred_prefers_b = score < 0
            target_prefers_b = target_score > 0.5
            if (pred_prefers_a and target_prefers_a) or (
                pred_prefers_b and target_prefers_b
            ):
                num_correct += 1
            num_samples += 1

        if only_consensus:
            title = "Final Accuracy (Only Consensus)"
        else:
            title = "Final Accuracy (All)"
        print(
            f"{title}: {num_correct}/{num_samples} = {num_correct / num_samples * 100:.1f}%"
        )

    await client.close()


if __name__ == "__main__":
    print(format_globals(globals()))
    asyncio.run(main())
