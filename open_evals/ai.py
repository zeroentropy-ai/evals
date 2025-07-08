import asyncio
import base64
import hashlib
import json
import os
import time
from collections.abc import Callable, Coroutine
from enum import Enum
from pathlib import Path
from typing import Any, Literal, TextIO, cast
from uuid import uuid4

import anthropic
import cohere
import cohere.core
import diskcache as dc  # pyright: ignore[reportMissingTypeStubs]
import httpx
import numpy as np
import openai
from openai import AsyncOpenAI
import redis.exceptions
import tiktoken
import torch
import voyageai
import voyageai.client_async
import voyageai.error
from anthropic import NOT_GIVEN, Anthropic, AsyncAnthropic, NotGiven
from anthropic.types import MessageParam
from dotenv import load_dotenv
from loguru import logger
from mxbai_rerank import MxbaiRerankV2 # pyright: ignore[reportMissingTypeStubs]
from openai.types.chat import ChatCompletionMessageParam
from openlimit.rate_limiters import (  # pyright: ignore[reportMissingTypeStubs]
    RateLimiter,
)
from openlimit.redis_rate_limiters import (  # pyright: ignore[reportMissingTypeStubs]
    RateLimiterWithRedis,
)
from pydantic import BaseModel, ValidationError, computed_field
from sentence_transformers import CrossEncoder, SentenceTransformer

from open_evals.utils import ROOT

load_dotenv(override=True)

REDIS_URL = None
AI_CACHE_DIR = f"{ROOT}/.cache"
AI_CACHE_SIZE_LIMIT = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEBUG = False

USING_VERTEX_AI = bool(os.environ.get("USING_VERTEX_AI"))

AIEmbedding = np.ndarray[Literal[1], np.dtype[np.float32]]

# AI Types


class AIModel(BaseModel):
    company: Literal["openai", "google", "anthropic"]
    model: str

    @computed_field
    @property
    def ratelimit_tpm(self) -> float:
        match self.company:
            case "openai":
                # Tier 5
                match self.model:
                    case _ if self.model.startswith("gpt-4o-mini"):
                        return 150_000_000
                    case _ if self.model.startswith("gpt-4o"):
                        return 30_000_000
                    case "gpt-4-turbo":
                        return 2_000_000
                    case _:
                        return 1_000_000
            case "google":
                if USING_VERTEX_AI:
                    return 50_000_000
                else:
                    # Tier 2
                    return 5_000_000
            case "anthropic":
                # Tier 4
                return 80_000

    @computed_field
    @property
    def ratelimit_rpm(self) -> float:
        match self.company:
            case "openai":
                # Tier 5
                match self.model:
                    case _ if self.model.startswith("gpt-4o-mini"):
                        return 30_000
                    case _:
                        return 10_000
            case "google":
                if USING_VERTEX_AI:
                    return 10_000
                else:
                    # Tier 2
                    return 1_000
            case "anthropic":
                # Tier 4
                return 4_000


class AIMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class AIEmbeddingModel(BaseModel):
    company: Literal["openai", "together", "cohere", "voyageai", "jina", "huggingface"]
    model: str

    @computed_field
    @property
    def dimensions(self) -> int:
        match self.company:
            case "openai":
                match self.model:
                    case "text-embedding-3-large":
                        return 3072
                    case "text-embedding-3-small":
                        return 1536
                    case _:
                        pass
            case "together":
                pass
            case "voyageai":
                match self.model:
                    case "voyage-law-2":
                        return 1024
                    case _:
                        pass
            case "cohere":
                pass
            case "jina":
                pass
            case "huggingface":
                pass
        raise NotImplementedError("Unknown Dimensions")

    @computed_field
    @property
    def ratelimit_tpm(self) -> float:
        match self.company:
            case "openai":
                return 10_000_000
            case "together":
                return 10_000_000
            case "jina":
                return 1_000_000
            case "cohere":
                return float("inf")
            case "voyageai":
                # Should be 1M, Manual Adjustment
                return 500_000
            case "huggingface":
                return 1_000_000

    @computed_field
    @property
    def ratelimit_rpm(self) -> float:
        match self.company:
            case "openai":
                return 10_000
            case "together":
                return 10_000
            case "jina":
                return 1_000
            case "cohere":
                return 1_000
            case "voyageai":
                # Manual Adjustment
                return 50
            case "huggingface":
                return 1_000

    @computed_field
    @property
    def max_batch_num_vectors(self) -> int:
        match self.company:
            case "openai":
                return 2048
            case "together":
                return 2048
            case "jina":
                return 1024
            case "cohere":
                return 96
            case "voyageai":
                return 128
            case "huggingface":
                return 1024

    @computed_field
    @property
    def max_batch_num_tokens(self) -> int:
        match self.company:
            case "openai":
                # 600k
                return 128_000
            case "jina":
                # ~80k
                return 50_000
            case "huggingface":
                return 1_000_000
            case _:
                return 100_000


class AIEmbeddingType(Enum):
    DOCUMENT = 1
    QUERY = 2


class AIRerankModel(BaseModel):
    company: Literal["cohere", "voyageai", "together", "jina", "huggingface"]
    model: str

    @computed_field
    @property
    def ratelimit_tpm(self) -> float:
        match self.company:
            case "voyageai":
                return 2_000_000
            case "cohere" | "jina" | "together" | "huggingface":
                return float("inf")

    @computed_field
    @property
    def ratelimit_rpm(self) -> float:
        match self.company:
            case "cohere":
                return 1000
            case "voyageai":
                # It says 100RPM but I can only get 60 out of it
                return 60
            case "together":
                return 1000
            case "jina":
                return 1000
            case "huggingface":
                return 1000


# Cache (Default=1GB, LRU)
os.makedirs(AI_CACHE_DIR, exist_ok=True)
g_cache: dc.Cache | None
if AI_CACHE_SIZE_LIMIT is None:
    g_cache = None
else:
    g_cache = dc.Cache(f"{AI_CACHE_DIR}/ai_cache.db", size_limit=AI_CACHE_SIZE_LIMIT)

RATE_LIMIT_RATIO = 0.95


class AIConnection:
    openai_client: AsyncOpenAI
    anthropic_client: AsyncAnthropic
    sync_anthropic_client: Anthropic
    google_client: AsyncOpenAI
    voyageai_client: voyageai.client_async.AsyncClient | None
    cohere_client: cohere.AsyncClient | None
    together_client: AsyncOpenAI
    together_rerank_client: cohere.AsyncClient | None
    jina_client: httpx.AsyncClient
    huggingface_client: tuple[
        dict[str, SentenceTransformer], dict[str, CrossEncoder | MxbaiRerankV2]
    ]
    # Mapping from (company, model) to RateLimiter
    rate_limiters: dict[str, RateLimiter | RateLimiterWithRedis]
    backoff_semaphores: dict[str, asyncio.Semaphore]
    redis_semaphores: dict[str, asyncio.Semaphore]

    def __init__(self) -> None:
        self.openai_client = AsyncOpenAI()
        self.google_client = AsyncOpenAI(
            base_url="https://aiplatform.googleapis.com/v1/projects/zeroentropy-435019/locations/global/endpoints/openapi"
            if USING_VERTEX_AI
            else "https://generativelanguage.googleapis.com/v1beta/",
            api_key=os.environ.get("GEMINI_API_KEY"),
        )
        self.anthropic_client = AsyncAnthropic()
        self.sync_anthropic_client = Anthropic()
        try:
            self.voyageai_client = voyageai.client_async.AsyncClient()
        except voyageai.error.AuthenticationError:
            self.voyageai_client = None
        try:
            self.cohere_client = cohere.AsyncClient()
        except cohere.core.api_error.ApiError:
            self.cohere_client = None
        self.together_client = AsyncOpenAI(
            base_url="https://api.together.xyz/v1/",
            api_key=os.environ.get("TOGETHER_API_KEY"),
        )
        try:
            self.together_rerank_client = cohere.AsyncClient(
                base_url="https://api.together.xyz/",
                api_key=os.environ.get("TOGETHER_API_KEY"),
            )
        except cohere.core.api_error.ApiError:
            self.together_rerank_client = None
        self.jina_client = httpx.AsyncClient(http2=True)
        self.huggingface_client = ({}, {})

        self.rate_limiters = {}
        self.backoff_semaphores = {}
        self.redis_semaphores = {}

    async def ai_wait_ratelimit(
        self,
        model: AIModel | AIEmbeddingModel | AIRerankModel,
        num_tokens: int,
        backoff: float | None = None,
    ) -> None:
        key = f"{model.__class__}:{model.company}:{model.model}"
        if key not in self.rate_limiters:
            if REDIS_URL is None:
                self.rate_limiters[key] = RateLimiter(
                    request_limit=model.ratelimit_rpm * RATE_LIMIT_RATIO,
                    token_limit=model.ratelimit_tpm * RATE_LIMIT_RATIO,
                    token_counter=None,
                    bucket_size_in_seconds=15,
                )
            else:
                self.rate_limiters[key] = RateLimiterWithRedis(
                    request_limit=model.ratelimit_rpm * RATE_LIMIT_RATIO,
                    token_limit=model.ratelimit_tpm * RATE_LIMIT_RATIO,
                    token_counter=None,
                    bucket_size_in_seconds=15,
                    redis_url=REDIS_URL,
                    bucket_key=key,
                )
            self.backoff_semaphores[key] = asyncio.Semaphore(1)
            # Prevent too many redis connections.
            self.redis_semaphores[key] = asyncio.Semaphore(100)
        if backoff is not None:
            async with self.backoff_semaphores[key]:
                await asyncio.sleep(backoff)

        for _redis_retry in range(10):
            try:
                async with self.redis_semaphores[key]:
                    await self.rate_limiters[key].wait_for_capacity(num_tokens)  # pyright: ignore[reportUnknownMemberType]
                break
            except redis.exceptions.LockError:
                logger.warning("redis.exceptions.LockError")
                await asyncio.sleep(0.05)
                continue
            except (ConnectionResetError, redis.exceptions.ConnectionError):
                logger.exception("Redis Exception")
                await asyncio.sleep(0.05)
                continue


# NOTE: API Clients cannot be called from multiple event loops,
# So every asyncio event loop needs its own API connection
ai_connections: dict[asyncio.AbstractEventLoop, AIConnection] = {}


def get_ai_connection() -> AIConnection:
    event_loop = asyncio.get_event_loop()
    if event_loop not in ai_connections:
        ai_connections[event_loop] = AIConnection()
    return ai_connections[event_loop]


class AIError(Exception):
    """A class for AI Task Errors"""


class AIValueError(AIError, ValueError):
    """A class for AI Value Errors"""


class AITimeoutError(AIError, TimeoutError):
    """A class for AI Task Timeout Errors"""


class AIRuntimeError(AIError, RuntimeError):
    """A class for AI Task Timeout Errors"""


def tiktoken_truncate_by_num_tokens(
    s: str,
    max_tokens: int,
    *,
    model: str = "cl100k_base",
) -> str:
    encoding = tiktoken.get_encoding(model)
    tokens = encoding.encode(s)
    tokens = tokens[:max_tokens]
    return encoding.decode(tokens)


def ai_num_tokens(model: AIModel | AIEmbeddingModel | AIRerankModel, s: str) -> int:
    if isinstance(model, AIModel):
        if model.company == "anthropic":
            # Doesn't actually connect to the network
            return (
                get_ai_connection()
                .sync_anthropic_client.messages.count_tokens(
                    model=model.model,
                    system="",
                    messages=[
                        {
                            "role": "user",
                            "content": s,
                        }
                    ],
                )
                .input_tokens
            )
        elif model.company == "openai":
            if model.model.startswith("gpt-4.1"):
                model_str = "gpt-4"
            else:
                model_str = model.model
            encoding = tiktoken.encoding_for_model(model_str)
            num_tokens = len(encoding.encode(s))
            return num_tokens
    if isinstance(model, AIEmbeddingModel):
        if model.company == "openai":
            encoding = tiktoken.encoding_for_model(model.model)
            num_tokens = len(encoding.encode(s))
            return num_tokens
        elif model.company == "voyageai":
            voyageai_client = get_ai_connection().voyageai_client
            if voyageai_client is not None:
                return voyageai_client.count_tokens([s], model.model)
            else:
                logger.exception("VoyageAI Client is not available")
    # Otherwise, estimate
    # logger.warning("Estimating Tokens!")
    return int(len(s) / 4)


def get_call_cache_key(
    model: AIModel,
    messages: list[AIMessage],
) -> str:
    # Hash the array of texts
    md5_hasher = hashlib.md5()
    md5_hasher.update(model.model_dump_json().encode())
    for message in messages:
        md5_hasher.update(md5_hasher.hexdigest().encode())
        md5_hasher.update(message.model_dump_json().encode())
    key = md5_hasher.hexdigest()

    return key


usage_files: dict[str, TextIO] = {}


def get_usage_file(filename: str) -> TextIO:
    if filename not in usage_files:
        path = Path(f"{ROOT}/logs/usage/{filename}.csv")
        os.makedirs(path.parent, exist_ok=True)
        usage_files[filename] = open(path, "a")  # noqa: SIM115
    return usage_files[filename]


async def ai_call[T: str | BaseModel](
    model: AIModel,
    messages: list[AIMessage],
    *,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    # When using anthropic, the first message must be from the user.
    # If the first message is not a User, this message will be prepended to the messages.
    anthropic_initial_message: str | None = "<START>",
    # If two messages of the same role are given to anthropic, they must be concatenated.
    # This is the delimiter between concatenated.
    anthropic_combine_delimiter: str = "\n",
    # Throw an AITimeoutError after this many retries fail
    num_ratelimit_retries: int = 10,
    # Backoff function (Receives index of attempt)
    backoff_algo: Callable[[int], float] = lambda i: min(2**i, 5),
    # The output type for the ai_call. Valid options are a pydantic BaseModel or a str. Using a BaseModel will use the Structured Output API.
    response_format: type[T] = str,
    # Usage Filename. If provided, will store usage statistics in ./logs/usage/{usage_filename}.csv
    usage_filename: str | None = None,
) -> T:
    cache_key = get_call_cache_key(model, messages)
    cached_call = cast(Any, g_cache.get(cache_key)) if g_cache is not None else None  # pyright: ignore[reportUnknownMemberType]

    if cached_call is not None:
        assert isinstance(cached_call, response_format)
        return cached_call

    num_tokens_input: int = sum(
        [ai_num_tokens(model, message.content) for message in messages]
    )

    return_value: T | None = None
    match model.company:
        case "openai" | "google":
            for i in range(num_ratelimit_retries):
                try:
                    await get_ai_connection().ai_wait_ratelimit(
                        model, num_tokens_input, backoff_algo(i - 1) if i > 0 else None
                    )

                    def ai_message_to_openai_message_param(
                        message: AIMessage,
                    ) -> ChatCompletionMessageParam:
                        if message.role == "system":  # noqa: SIM114
                            return {"role": message.role, "content": message.content}
                        elif message.role == "user":  # noqa: SIM114
                            return {"role": message.role, "content": message.content}
                        elif message.role == "assistant":
                            return {"role": message.role, "content": message.content}
                        raise NotImplementedError("Unreachable Code")

                    if i > 0 and DEBUG:
                        logger.debug("Trying again after RateLimitError...")
                    match model.company:
                        case "openai":
                            client = get_ai_connection().openai_client
                            model_str = model.model
                            extra_body = None
                        case "google":
                            client = get_ai_connection().google_client
                            extra_body = {
                                "extra_body": {
                                    "google": {
                                        "thinking_config": {
                                            "thinking_budget": 0,
                                        }
                                    }
                                }
                            }
                            model_str = model.model
                            if USING_VERTEX_AI:
                                model_str = f"google/{model_str}"
                    if response_format is str:
                        response = await client.chat.completions.create(
                            model=model_str,
                            messages=[
                                ai_message_to_openai_message_param(message)
                                for message in messages
                            ],
                            temperature=temperature,
                            max_tokens=max_tokens,
                            extra_body=extra_body,
                        )
                        response_content = response.choices[0].message.content
                        assert response_content is not None
                        assert isinstance(response_content, response_format)
                        return_value = response_content
                    else:
                        response = await client.beta.chat.completions.parse(
                            model=model_str,
                            messages=[
                                ai_message_to_openai_message_param(message)
                                for message in messages
                            ],
                            temperature=0,
                            max_tokens=max_tokens,
                            response_format=response_format,
                            extra_body=extra_body,
                        )
                        assert response.usage is not None
                        reasoning_tokens = (
                            response.usage.completion_tokens_details.reasoning_tokens
                            if response.usage.completion_tokens_details is not None
                            else None
                        )
                        if usage_filename is not None:
                            f = get_usage_file(usage_filename)
                            f.write(
                                f"{response.usage.prompt_tokens},{response.usage.completion_tokens},{reasoning_tokens}\n"
                            )
                            f.flush()
                        response_parsed = response.choices[0].message.parsed
                        assert response_parsed is not None
                        assert isinstance(response_parsed, response_format)
                        return_value = response_parsed
                    break
                except openai.LengthFinishReasonError as e:
                    raise AIRuntimeError("LengthFinishReasonError") from e
                except openai.RateLimitError as e:
                    logger.warning(f"OpenAI RateLimitError: {e}")
            if return_value is None:
                raise AITimeoutError("Cannot overcome OpenAI RateLimitError")

        case "anthropic":
            for i in range(num_ratelimit_retries):
                try:
                    await get_ai_connection().ai_wait_ratelimit(
                        model, num_tokens_input, backoff_algo(i - 1) if i > 0 else None
                    )

                    def ai_message_to_anthropic_message_param(
                        message: AIMessage,
                    ) -> MessageParam:
                        if message.role == "user" or message.role == "assistant":
                            return {"role": message.role, "content": message.content}
                        elif message.role == "system":
                            raise AIValueError(
                                "system not allowed in anthropic message param"
                            )
                        raise NotImplementedError("Unreachable Code")

                    if i > 0 and DEBUG:
                        logger.debug("Trying again after RateLimitError...")

                    # Extract system message if it exists
                    system: str | NotGiven = NOT_GIVEN
                    if len(messages) > 0 and messages[0].role == "system":
                        system = messages[0].content
                        messages = messages[1:]
                    if issubclass(response_format, BaseModel):
                        if not isinstance(system, str):
                            system = ""
                        system = (
                            f"Please respond with a JSON object adhering to the provided JSON schema. Don't provide any extra fields, and don't respond with anything other than the JSON object.\n\n{json.dumps(response_format.model_json_schema())}"
                            + system
                        )
                        messages.append(
                            AIMessage(
                                role="assistant",
                                content="```json",
                            )
                        )

                    # Insert initial message if necessary
                    if (
                        anthropic_initial_message is not None
                        and len(messages) > 0
                        and messages[0].role != "user"
                    ):
                        messages = [
                            AIMessage(role="user", content=anthropic_initial_message)
                        ] + messages
                    # Combined messages (By combining consecutive messages of the same role)
                    combined_messages: list[AIMessage] = []
                    for message in messages:
                        if (
                            len(combined_messages) == 0
                            or combined_messages[-1].role != message.role
                        ):
                            combined_messages.append(message)
                        else:
                            # Copy before edit
                            combined_messages[-1] = combined_messages[-1].model_copy(
                                deep=True
                            )
                            # Merge consecutive messages with the same role
                            combined_messages[-1].content += (
                                anthropic_combine_delimiter + message.content
                            )
                    # Get the response
                    response_message = (
                        await get_ai_connection().anthropic_client.messages.create(
                            model=model.model,
                            system=system,
                            messages=[
                                ai_message_to_anthropic_message_param(message)
                                for message in combined_messages
                            ],
                            temperature=0.0,
                            max_tokens=max_tokens,
                        )
                    )
                    assert isinstance(
                        response_message.content[0], anthropic.types.TextBlock
                    )
                    response_content = response_message.content[0].text
                    assert isinstance(response_content, str)
                    if response_format is str:
                        assert isinstance(response_content, response_format)
                        return_value = response_content  # pyright: ignore[reportAssignmentType]
                    else:
                        assert issubclass(response_format, BaseModel)
                        response_content = response_content.strip()
                        if response_content.startswith("```json"):
                            response_content = response_content[len("```json") :]
                        while True:
                            if response_content.endswith("```"):
                                response_content = response_content[: -len("```")]
                                response_content = response_content.strip()
                            else:
                                break
                        try:
                            return_value = response_format.model_validate_json(
                                response_content
                            )
                        except ValidationError as e:
                            print(f"Invalid: {response_content}")
                            raise AIRuntimeError(
                                "Failed to Validate Response Content."
                            ) from e
                    break
                except (anthropic.RateLimitError, anthropic.BadRequestError) as e:
                    logger.warning(f"Anthropic Error: {repr(e)}")
            if return_value is None:
                raise AITimeoutError("Cannot overcome Anthropic RateLimitError")

    if g_cache is not None:
        g_cache.set(cache_key, return_value)  # pyright: ignore[reportUnknownMemberType]
    return return_value


def get_embeddings_cache_key(
    model: AIEmbeddingModel, text: str, embedding_type: AIEmbeddingType
) -> str:
    key = f"{model.company}||||{model.model}||||{embedding_type.name}||||{hashlib.md5(text.encode()).hexdigest()}"
    return key


def cosine_similarity(vec1: AIEmbedding, vec2: AIEmbedding) -> float:
    return np.dot(vec1, vec2)


async def ai_embedding(
    model: AIEmbeddingModel,
    texts: list[str],
    embedding_type: AIEmbeddingType,
    *,
    # Throw an AITimeoutError after this many retries fail
    num_ratelimit_retries: int = 10,
    # Backoff function (Receives index of attempt)
    backoff_algo: Callable[[int], float] = lambda i: min(2**i, 5),
    # Callback (For tracking progress)
    callback: Callable[[], Any] = lambda: None,
    # Cache
    cache: dc.Cache | None = None,
    # Num Tokens (Internal: To prevent recalculating)
    _texts_num_tokens: list[int] | None = None,
) -> list[AIEmbedding]:
    if cache is None:
        cache = g_cache
    if _texts_num_tokens is None:
        _texts_num_tokens = [ai_num_tokens(model, text) for text in texts]

    # Extract cache miss indices
    text_embeddings: list[AIEmbedding | None] = [None] * len(texts)
    start_time = time.time()
    if cache is not None:
        with cache.transact():
            for i, text in enumerate(texts):
                cache_key = get_embeddings_cache_key(model, text, embedding_type)
                cache_result = cast(Any, cache.get(cache_key))  # pyright: ignore[reportUnknownMemberType]
                if cache_result is not None:
                    if not isinstance(cache_result, np.ndarray):
                        logger.warning("Invalid cache_result, ignoring...")
                        continue
                    callback()
                    cache_result = cast(AIEmbedding, cache_result)
                    text_embeddings[i] = cache_result
        end_time = time.time()
        if DEBUG:
            logger.debug(f"Cache Read Time: {(end_time - start_time) * 1000:.2f}ms")
    if not any(embedding is None for embedding in text_embeddings):
        return cast(list[AIEmbedding], text_embeddings)
    required_text_embeddings_indices = [
        i for i in range(len(text_embeddings)) if text_embeddings[i] is None
    ]

    num_tokens_input: int = sum(
        [_texts_num_tokens[index] for index in required_text_embeddings_indices]
    )

    # Recursively Batch if necessary
    if len(required_text_embeddings_indices) > model.max_batch_num_vectors or (
        num_tokens_input > model.max_batch_num_tokens
        and len(required_text_embeddings_indices) > 1
    ):
        # Calculate batch size
        batch_size = model.max_batch_num_vectors
        # If we wouldn't split on the basis of max batch num vectors, but we should based on tokens, then we'll lower the batch size
        if (
            len(required_text_embeddings_indices) <= model.max_batch_num_vectors
            and num_tokens_input > model.max_batch_num_tokens
        ):
            batch_size = max(len(required_text_embeddings_indices) // 2, 1)

        # Calculate embeddings in batches
        tasks: list[Coroutine[Any, Any, list[AIEmbedding]]] = []
        for i in range(0, len(required_text_embeddings_indices), batch_size):
            batch_indices = required_text_embeddings_indices[i : i + batch_size]
            tasks.append(
                ai_embedding(
                    model,
                    [texts[i] for i in batch_indices],
                    embedding_type,
                    num_ratelimit_retries=num_ratelimit_retries,
                    backoff_algo=backoff_algo,
                    callback=callback,
                    cache=cache,
                    _texts_num_tokens=[_texts_num_tokens[i] for i in batch_indices],
                )
            )
        preflattened_results = await asyncio.gather(*tasks)
        results: list[AIEmbedding] = []
        for embeddings_list in preflattened_results:
            results.extend(embeddings_list)
        # Merge with cache hits
        assert len(required_text_embeddings_indices) == len(results)
        for i, embedding in zip(required_text_embeddings_indices, results, strict=True):
            text_embeddings[i] = embedding
        assert all(embedding is not None for embedding in text_embeddings)
        return cast(list[AIEmbedding], text_embeddings)

    input_texts = [texts[i] for i in required_text_embeddings_indices]
    text_embeddings_response: list[AIEmbedding] | None = None
    match model.company:
        case "openai" | "together":
            for i in range(num_ratelimit_retries):
                try:
                    call_id = uuid4()
                    if DEBUG:
                        logger.debug(
                            f"Start AIEmbedding Call {call_id} (N_TOKENS={num_tokens_input})"
                        )
                    start_time = time.time()
                    await get_ai_connection().ai_wait_ratelimit(
                        model, num_tokens_input, backoff_algo(i - 1) if i > 0 else None
                    )
                    end_time = time.time()
                    if DEBUG:
                        logger.debug(
                            f"AIEmbedding RateLimit Wait Time {call_id}: {(end_time - start_time) * 1000:.2f}ms (N_TOKENS={num_tokens_input})"
                        )
                    prepared_input_texts = [text for text in input_texts]
                    for i, text in enumerate(prepared_input_texts):
                        if len(text) == 0:
                            prepared_input_texts[i] = " "
                    if model.model.startswith("BAAI/"):
                        for i, _text in enumerate(prepared_input_texts):
                            prepared_input_texts[i] = prepared_input_texts[i][:1024]
                    start_time = time.time()
                    match model.company:
                        case "openai":
                            client = get_ai_connection().openai_client
                            using_base64 = True
                        case "together":
                            client = get_ai_connection().together_client
                            using_base64 = False
                    response = await client.embeddings.create(
                        input=prepared_input_texts,
                        model=model.model,
                        encoding_format="base64" if using_base64 else "float",
                    )
                    end_time = time.time()
                    if DEBUG:
                        logger.debug(
                            f"AIEmbedding Call {call_id}: {(end_time - start_time) * 1000:.2f}ms (N_TOKENS={num_tokens_input})"
                        )
                    response_embeddings: list[AIEmbedding] = []
                    for embedding_obj in response.data:
                        if using_base64:
                            data = cast(object, embedding_obj.embedding)
                            if not isinstance(data, str):
                                # numpy is not installed / base64 optimisation isn't enabled for this model yet
                                raise RuntimeError("Error with base64/numpy")

                            response_embeddings.append(
                                np.frombuffer(base64.b64decode(data), dtype="float32")
                            )
                        else:
                            response_embeddings.append(
                                np.array(embedding_obj.embedding)
                            )
                    t2 = time.time()
                    if DEBUG:
                        logger.debug(
                            f"AIEmbedding Decoding Time {call_id}: {(t2 - end_time) * 1000:.2f}ms"
                        )
                    text_embeddings_response = response_embeddings
                    break
                except openai.BadRequestError:
                    raise
                except (
                    openai.RateLimitError,
                    openai.APITimeoutError,
                ):
                    if DEBUG:
                        logger.warning("AIEmbedding RateLimitError")
                except openai.APIError as e:
                    logger.exception(f"AIEmbedding Unknown Error: {repr(e)}")
            if text_embeddings_response is None:
                raise AITimeoutError("Cannot overcome OpenAI RateLimitError")
        case "cohere":
            for i in range(num_ratelimit_retries):
                try:
                    await get_ai_connection().ai_wait_ratelimit(
                        model, num_tokens_input, backoff_algo(i - 1) if i > 0 else None
                    )
                    cohere_client = get_ai_connection().cohere_client
                    if cohere_client is None:
                        raise AIValueError("Cohere Credentials are not available")
                    result = await cohere_client.embed(
                        texts=input_texts,
                        model=model.model,
                        input_type=(
                            "search_document"
                            if embedding_type == AIEmbeddingType.DOCUMENT
                            else "search_query"
                        ),
                    )
                    assert isinstance(result.embeddings, list)
                    text_embeddings_response = [
                        np.array(embedding) for embedding in result.embeddings
                    ]
                    break
                except (
                    cohere.errors.TooManyRequestsError,
                    httpx.ConnectError,
                    httpx.RemoteProtocolError,
                    httpx.ReadError,
                ):
                    logger.warning("Cohere RateLimitError")
            if text_embeddings_response is None:
                raise AITimeoutError("Cannot overcome Cohere RateLimitError")
        case "voyageai":
            for i in range(num_ratelimit_retries):
                try:
                    await get_ai_connection().ai_wait_ratelimit(
                        model, num_tokens_input, backoff_algo(i - 1) if i > 0 else None
                    )
                    voyageai_client = get_ai_connection().voyageai_client
                    if voyageai_client is None:
                        raise AIValueError("VoyageAI Credentials are not available")
                    result = await voyageai_client.embed(
                        input_texts,
                        model=model.model,
                        input_type=(
                            "document"
                            if embedding_type == AIEmbeddingType.DOCUMENT
                            else "query"
                        ),
                    )
                    assert isinstance(result.embeddings, list)
                    text_embeddings_response = [
                        np.array(embedding) for embedding in result.embeddings
                    ]
                    break
                except (
                    voyageai.error.RateLimitError,
                    voyageai.error.APIConnectionError,
                ):
                    logger.warning("VoyageAI RateLimitError")
            if text_embeddings_response is None:
                raise AITimeoutError("Cannot overcome VoyageAI RateLimitError")
        case "jina":
            task_id = uuid4()
            for i in range(num_ratelimit_retries):
                try:
                    call_id = uuid4()
                    if DEBUG:
                        logger.debug(
                            f"Start AIEmbedding Call {call_id} (task={task_id})"
                        )
                    start_time = time.time()
                    await get_ai_connection().ai_wait_ratelimit(
                        model, num_tokens_input, backoff_algo(i - 1) if i > 0 else None
                    )
                    end_time = time.time()
                    if DEBUG:
                        logger.debug(
                            f"AIEmbedding RateLimit Wait Time {call_id}: {(end_time - start_time) * 1000:.2f}ms (N_TOKENS={num_tokens_input})"
                        )
                    start_time = time.time()
                    match embedding_type:
                        case AIEmbeddingType.QUERY:
                            task_str = "retrieval.query"
                        case AIEmbeddingType.DOCUMENT:
                            task_str = "retrieval.passage"
                    response = await get_ai_connection().jina_client.post(
                        "https://api.jina.ai/v1/embeddings",
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {os.environ['JINA_API_KEY']}",
                        },
                        json={
                            "model": model.model,
                            "task": task_str,
                            "input": input_texts,
                        },
                    )
                    response.raise_for_status()
                    response = response.json()

                    end_time = time.time()
                    if DEBUG:
                        logger.debug(
                            f"AIEmbedding Call {call_id}: {(end_time - start_time) * 1000:.2f}ms (N_TOKENS={num_tokens_input})"
                        )
                    response_embeddings_jina: list[AIEmbedding] = []
                    if "data" not in response:
                        raise ValueError(
                            "Response missing data: " + json.dumps(response, indent=4)
                        )
                    for embedding_obj in response["data"]:
                        assert embedding_obj["object"] == "embedding"
                        assert isinstance(embedding_obj["embedding"], list)
                        response_embeddings_jina.append(
                            np.array(embedding_obj["embedding"])
                        )
                    t2 = time.time()
                    if DEBUG:
                        logger.debug(
                            f"AIEmbedding Decoding Time {call_id}: {(t2 - end_time) * 1000:.2f}ms"
                        )
                    text_embeddings_response = response_embeddings_jina
                    break
                except (
                    httpx.ConnectError,
                    httpx.RemoteProtocolError,
                    httpx.ReadError,
                    httpx.ReadTimeout,
                    httpx.ConnectTimeout,
                ):
                    logger.warning("JinaAI RateLimitError")
            if text_embeddings_response is None:
                raise AITimeoutError("Cannot overcome JinaAI RateLimitError")
        case "huggingface":
            model_name = model.model
            hf_sentence_transformers = get_ai_connection().huggingface_client[0]
            if model_name not in hf_sentence_transformers:
                hf_sentence_transformers[model_name] = SentenceTransformer(
                    model_name,
                    device=DEVICE,
                )
            text_embeddings_response = [
                cast(AIEmbedding, embedding)
                for embedding in hf_sentence_transformers[model_name].encode(  # pyright: ignore[reportUnknownMemberType]
                    input_texts,
                    normalize_embeddings=True,
                )
            ]

    # Update cache
    if cache is not None:
        start_time = time.time()
        with cache.transact():
            assert len(text_embeddings_response) == len(
                required_text_embeddings_indices
            )
            for index, embedding in zip(
                required_text_embeddings_indices, text_embeddings_response, strict=True
            ):
                cache_key = get_embeddings_cache_key(
                    model, texts[index], embedding_type
                )
                cache.set(cache_key, embedding)  # pyright: ignore[reportUnknownMemberType]
        end_time = time.time()
        if DEBUG:
            logger.debug(f"Cache Write Time: {(end_time - start_time) * 1000:.2f}ms")
    for index, embedding in zip(
        required_text_embeddings_indices, text_embeddings_response, strict=True
    ):
        text_embeddings[index] = embedding
        callback()
    assert all(embedding is not None for embedding in text_embeddings)
    return cast(list[AIEmbedding], text_embeddings)


def get_rerank_cache_key(model: AIRerankModel, query: str, text: str) -> str:
    # input
    md5_hasher = hashlib.md5()
    md5_hasher.update(model.model_dump_json().encode())
    md5_hasher.update(query.encode())
    md5_hasher.update(md5_hasher.hexdigest().encode())
    md5_hasher.update(text.encode())
    texts_hash = md5_hasher.hexdigest()
    return texts_hash


# Gets the list of indices that reranks the original texts
async def ai_rerank(
    model: AIRerankModel,
    query: str,
    texts: list[str],
    *,
    top_k: int | None = None,
    # Throw an AITimeoutError after this many retries fail
    num_ratelimit_retries: int = 10,
    # Backoff function (Receives index of attempt)
    backoff_algo: Callable[[int], float] = lambda i: min(2**i, 5),
) -> list[float]:
    text_scores: list[float | None] = [None] * len(texts)
    if g_cache is not None:
        for i, text in enumerate(texts):
            cache_key = get_rerank_cache_key(model, query, text)
            cache_result = cast(Any, g_cache.get(cache_key))  # pyright: ignore[reportUnknownMemberType]
            if cache_result is not None:
                # cast instead of assert isinstance, because of ints
                cache_result = float(cache_result)
                text_scores[i] = cache_result
    if all(score is not None for score in text_scores):
        return cast(list[float], text_scores)

    unprocessed_indices = [i for i, score in enumerate(text_scores) if score is None]
    unprocessed_texts = [texts[i] for i in unprocessed_indices]
    num_tokens_input = sum(ai_num_tokens(model, text) for text in unprocessed_texts)

    relevance_scores: list[float] | None = None
    match model.company:
        case "cohere" | "together":
            for i in range(num_ratelimit_retries):
                try:
                    await get_ai_connection().ai_wait_ratelimit(
                        model, num_tokens_input, backoff_algo(i - 1) if i > 0 else None
                    )
                    match model.company:
                        case "cohere":
                            cohere_client = get_ai_connection().cohere_client
                        case "together":
                            cohere_client = get_ai_connection().together_rerank_client
                            if top_k is None:  # Together doesn't accept null for top_k
                                top_k = len(unprocessed_texts)
                    if cohere_client is None:
                        raise AIValueError(
                            f"{model.company.capitalize()} Credentials are not available"
                        )
                    response = await cohere_client.rerank(
                        model=model.model,
                        query=query,
                        documents=unprocessed_texts,
                        top_n=top_k,
                    )
                    original_order_results = sorted(
                        response.results, key=lambda x: x.index
                    )
                    relevance_scores = [
                        result.relevance_score for result in original_order_results
                    ]
                    break
                except (
                    cohere.errors.TooManyRequestsError,
                    httpx.ConnectError,
                    httpx.RemoteProtocolError,
                ):
                    logger.warning(f"{model.company.capitalize()} RateLimitError")
                except cohere.errors.bad_request_error.BadRequestError:
                    logger.exception(
                        f"{model.company.capitalize()} had BadRequestError"
                    )
                    raise
            if relevance_scores is None:
                raise AITimeoutError(
                    f"Cannot overcome {model.company.capitalize()} RateLimitError"
                )
        case "voyageai":
            for i in range(num_ratelimit_retries):
                try:
                    await get_ai_connection().ai_wait_ratelimit(
                        model, num_tokens_input, backoff_algo(i - 1) if i > 0 else None
                    )
                    voyageai_client = get_ai_connection().voyageai_client
                    if voyageai_client is None:
                        raise AIValueError("VoyageAI Credentials are not available")
                    voyageai_response = await voyageai_client.rerank(
                        query=query,
                        documents=unprocessed_texts,
                        model=model.model,
                        top_k=top_k,
                    )
                    original_order_results = sorted(
                        voyageai_response.results,
                        key=lambda x: int(x.index),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                    )
                    relevance_scores = [
                        float(result.relevance_score)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                        for result in original_order_results
                    ]
                    break
                except voyageai.error.RateLimitError:
                    logger.warning("VoyageAI RateLimitError")
            if relevance_scores is None:
                raise AITimeoutError("Cannot overcome VoyageAI RateLimitError")
        case "jina":
            for i in range(num_ratelimit_retries):
                try:
                    await get_ai_connection().ai_wait_ratelimit(
                        model, num_tokens_input, backoff_algo(i - 1) if i > 0 else None
                    )
                    response = await get_ai_connection().jina_client.post(
                        "https://api.jina.ai/v1/rerank",
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {os.environ['JINA_API_KEY']}",
                        },
                        json={
                            "model": model.model,
                            "query": query,
                            "documents": unprocessed_texts,
                            "return_documents": False,
                        },
                    )
                    response.raise_for_status()
                    response = response.json()
                    original_order_results = sorted(
                        response["results"], key=lambda x: x["index"]
                    )
                    relevance_scores = [
                        float(result["relevance_score"])
                        for result in original_order_results
                    ]
                    break
                except (
                    httpx.HTTPStatusError,
                    httpx.ConnectError,
                    httpx.RemoteProtocolError,
                    httpx.ReadTimeout,
                    httpx.ConnectTimeout,
                ):
                    logger.warning("JinaAI RateLimitError")
            if relevance_scores is None:
                raise AITimeoutError("Cannot overcome Cohere RateLimitError")
        case "huggingface":
            model_name = model.model
            hf_cross_encoders = get_ai_connection().huggingface_client[1]

            # Initialize the model if it hasn't been already
            if model_name not in hf_cross_encoders:
                if model_name.startswith("mixedbread-ai/"):
                    hf_cross_encoders[model_name] = MxbaiRerankV2(
                        model_name,
                        device=DEVICE,
                    )
                else: # This should work for Qwen/Qwen3-Reranker-4B etc.
                    hf_cross_encoders[model_name] = CrossEncoder(
                        model_name,
                        device=DEVICE,
                    )

            # Inference the model
            hf_cross_encoder = hf_cross_encoders[model_name]
            match hf_cross_encoder:
                case CrossEncoder():
                    scores = hf_cross_encoder.predict(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                        [(query, document) for document in unprocessed_texts]
                    )
                    relevance_scores = [float(score) for score in scores]
                case MxbaiRerankV2():
                    results = hf_cross_encoder.rank(  # pyright: ignore[reportUnknownMemberType]
                        query=query,
                        documents=unprocessed_texts,
                    )
                    results.sort(key=lambda x: x.index)
                    relevance_scores = [result.score for result in results]
        case "modal":
            raise NotImplementedError("Modal is not supported yet")

    assert len(unprocessed_indices) == len(relevance_scores)
    for index, score in zip(unprocessed_indices, relevance_scores, strict=True):
        if g_cache is not None:
            cache_key = get_rerank_cache_key(model, query, texts[index])
            g_cache.set(cache_key, score)  # pyright: ignore[reportUnknownMemberType]
        text_scores[index] = score

    assert all(score is not None for score in text_scores)
    return cast(list[float], text_scores)
