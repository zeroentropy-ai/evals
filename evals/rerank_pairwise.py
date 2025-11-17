import asyncio
import json
import os
from collections import defaultdict
from random import Random
from typing import Literal

import diskcache as dc  # pyright: ignore[reportMissingTypeStubs]
import httpx
import numpy as np
from pydantic import BaseModel

from evals.utils import async_zip, flatten, get_client, hash_str, npf

MODAL_MAX_SIMULTANEOUS_RERANKS = 20
MODAL_RERANK_SEMAPHORES: dict[str, asyncio.Semaphore] = defaultdict(
    lambda: asyncio.Semaphore(MODAL_MAX_SIMULTANEOUS_RERANKS)
)
MODAL_MAX_SIMULTANEOUS_REQUESTS = 250
MODAL_REQUEST_SEMAPHORES: dict[str, asyncio.Semaphore] = defaultdict(
    lambda: asyncio.Semaphore(MODAL_MAX_SIMULTANEOUS_REQUESTS)
)
MODAL_BATCH_SIZE_TOKENS = 100_000


def sample_from_random_cycles(n: int, k: int, *, rng: Random) -> list[tuple[int, int]]:
    selected_pair_indices_set: set[tuple[int, int]] = set()
    for _cycle in range(n):
        if len(selected_pair_indices_set) > k:
            break
        indices = list(range(n))
        rng.shuffle(indices)
        for index in range(n):
            if len(selected_pair_indices_set) > k:
                break
            i, j = indices[index], indices[(index + 1) % n]
            i2, j2 = sorted([i, j])
            key = (i2, j2)
            if key in selected_pair_indices_set:
                continue
            selected_pair_indices_set.add(key)

    selected_pair_indices = list(selected_pair_indices_set)
    return selected_pair_indices


def elos_loss(w: npf, elos: npf) -> float:
    N = len(elos)
    elos_col = elos.reshape(-1, 1)
    elos_row = elos.reshape(1, -1)

    # Stable computation of log(exp(elos_i) + exp(elos_j))
    max_elos = np.maximum(elos_col, elos_row)
    log_pairwise_sums = max_elos + np.log(
        np.exp(elos_col - max_elos) + np.exp(elos_row - max_elos)
    )

    # Calculate elos_i - log(exp(elos_i) + exp(elos_j))
    log_diff = np.broadcast_to(elos_col, (N, N)) - log_pairwise_sums

    # We want to maximize the loglikelihood of the observed w with respect to elos
    loglikelihood = float(np.sum(w * log_diff))

    # Return the loss that we're trying to minimize
    return -loglikelihood


def calculate_elos(
    w: npf,
    *,
    # How close we must be to the log-likelihood loss
    epsilon: float = 1e-4,
    # If you have ELOs calculated from a similar W, then it will converge faster by initializing to the same ELOs
    initial_elos: npf | None = None,
    # Max iters before giving up
    max_iters: int = 1000,
) -> tuple[npf, list[float]]:
    # https://hackmd.io/@-Gjw1zWMSH6lMPRlziQFEw/B15B4Rsleg

    N = len(w)
    elos = initial_elos.copy() if initial_elos is not None else np.zeros(N)

    losses: list[float] = []
    for _iter in range(max_iters):
        # Create all pairwise differences elo_j - elo_i in a matrix
        # outer(ones, elos) - outer(elos, ones)
        D: npf = elos.reshape(1, N) - elos.reshape(N, 1)  # Shape: (N, N)

        # Calculate sigmoid matrix
        S: npf = 1.0 / (1.0 + np.exp(-D))  # S[i,j] = sigmoid(elo_j - elo_i)

        # Calculate the update terms
        numerator: npf = np.sum(w * S, axis=1)  # Shape: (N,)
        denominator: npf = np.sum(w.T * S.T, axis=1)  # Shape: (N,)

        # Apply update rule, using decreasing learning rate.
        learning_rate = float((1.0 + _iter) ** (-0.125))
        elos += (np.log(numerator) - np.log(denominator)) * learning_rate
        elos -= np.mean(elos)

        # Calculate loss for this iteration
        loss = elos_loss(w, elos)
        losses.append(loss)
        if len(losses) > 2 and abs(losses[-2] - losses[-1]) < epsilon:
            break
    if abs(losses[-2] - losses[-1]) > epsilon:
        print(f"ERROR! Not within epsilon after {len(losses)} iterations!")

    return elos, losses


class AIPairwiseReranker(BaseModel):
    company: Literal["modal"]
    model: str


class PairwiseResult(BaseModel):
    score: float
    score_logit: float
    kappa: float
    kappa_logit: float


class PairwiseOutput(BaseModel):
    results: list[PairwiseResult]


async def modal_pairwise_score(
    query_d1_d2: list[tuple[str, str, str]],
    *,
    url: str | None = None,
    timeout: float | None = None,
    num_retries: int = 7,
    cache: dc.Cache | None = None,
) -> PairwiseOutput:
    if url is None:
        url = "https://zeroentropy--ze-pairwise-rerank-v0-3-4-model-endpoint.modal.run/"
    if timeout is None:
        timeout = 180.0

    result = None
    for _retry in range(num_retries):
        try:
            async with MODAL_REQUEST_SEMAPHORES[url]:
                response = await get_client().post(
                    url,
                    headers={
                        "Modal-Key": os.environ["MODAL_KEY"],
                        "Modal-Secret": os.environ["MODAL_SECRET"],
                    },
                    json={
                        "query_d1_d2": query_d1_d2,
                    },
                    timeout=timeout,
                )
                response.raise_for_status()
                response_json = response.json()
            result = PairwiseOutput.model_validate(response_json)
            break  # Exit Loop
        except (
            httpx.ConnectError,
            httpx.RemoteProtocolError,
            httpx.ReadError,
            TimeoutError,
        ) as e:
            delay = 0.25 * (2**_retry)
            print(f"Request timed out: {e}. Retrying in {delay}s...")
            await asyncio.sleep(delay)
    if result is None:
        raise TimeoutError("Could not recover from API errors")

    return result


async def rerank_by_ai_pairwise_reranker(
    model: AIPairwiseReranker,
    query: str,
    documents: list[str],
    *,
    num_cycles: int,
    # Cache
    cache: dc.Cache | None = None,
) -> list[float]:
    num_docs = len(documents)
    if num_docs == 0:
        return []
    elif num_docs == 1:
        return [0]

    selected_pairs = sample_from_random_cycles(
        num_docs,
        num_docs * num_cycles,
        rng=Random(
            hash_str(
                json.dumps(
                    {
                        "model": model.model_dump(),
                        "query": query,
                        "documents": documents,
                    }
                )
            )
        ),
    )

    assert model.company == "modal"
    url = model.model
    async with MODAL_RERANK_SEMAPHORES[url]:
        # Format the selected pairs into batches
        batches: list[list[tuple[int, int, str, str]]] = []
        cumulative_length = 0
        for i, j in selected_pairs:
            document_a = documents[i]
            document_b = documents[j]

            if len(batches) == 0 or cumulative_length > MODAL_BATCH_SIZE_TOKENS:
                batches.append([])
                cumulative_length = 0

            batches[-1].append((i, j, document_a, document_b))
            cumulative_length += 20 + len(query) + len(document_a) + len(document_b)

        # Inference all of the pairs in this line
        all_outputs = await asyncio.gather(
            *[
                async_zip(
                    modal_pairwise_score([(query, d1, d2) for _i, _j, d1, d2 in batch]),
                    [(i, j) for i, j, _d1, _d2 in batch],
                )
                for batch in batches
            ]
        )

    all_results = [
        [(index, result) for result, index in zip(output.results, indices, strict=True)]
        for output, indices in all_outputs
    ]
    all_results = flatten(all_results)

    # Construct the empricial w, and then calculate ELOs
    w = np.zeros((num_docs, num_docs))
    np.fill_diagonal(w, 0.5)
    for (i, j), result in all_results:
        w[i][j] += result.score
        w[j][i] += 1.0 - result.score
    elos, _ = calculate_elos(w)
    elos = [float(elo) for elo in elos]

    return elos
