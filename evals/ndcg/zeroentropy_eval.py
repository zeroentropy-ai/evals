import aiohttp
from tqdm.asyncio import tqdm
from evals.utils import ROOT
# pyright: reportUnknownMemberType=false
DATASET_PATH = f"{ROOT}/data/datasets/bioasq/ai_scores.json"
USE_CLIENT = True
async def rerank_batch(query_documents: list[tuple[str, str]], client) -> list[float]:
    url = "https://npip99--testing-ze-rerank-1-model-endpoint.modal.run"
    headers = {
        "Modal-Key": "wk-zW97GGIGkt8og8BnVkt3XZ",
        "Modal-Secret": "ws-OiTrRmxnOjlV5tGfv9Yb3P",
    }
    payload = {
        "query_documents": query_documents,
    }
    if USE_CLIENT:
        assert client is not None
        async with client.post(url, headers=headers, json=payload) as response:
            result = await response.json()
            scores = [float(score) for score in result["scores"]]
    assert len(scores) == len(query_documents)
    return scores
