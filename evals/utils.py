import asyncio
import fcntl
import hashlib
import json
import math
import os
import sys
from collections.abc import Awaitable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any

import httpx
import numpy as np
from datasets import (  # pyright: ignore[reportMissingTypeStubs]
    Dataset,
    DatasetDict,
    load_from_disk,
)
from dotenv import load_dotenv
from numpy.typing import NDArray
from tqdm import tqdm

load_dotenv(override=True)


ROOT = f"{Path(__file__).resolve().parent.parent}"

# pyright: reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownLambdaType=false

# NumPy Float N-Dimensional Array
type npf = NDArray[np.float64]


def format_globals(globals_dict: dict[str, Any]) -> str:
    serializable_globals: dict[str, Any] = {
        "COMMAND": sys.argv[0],
    }
    if len(sys.argv) > 1:
        serializable_globals["ARGS"] = sys.argv[1:]
    for k, v in globals_dict.items():
        if k.upper() != k or k in ["ROOT"]:
            continue
        try:
            json.dumps(v)
            serializable_globals[k] = v
        except (TypeError, OverflowError):
            pass
    return json.dumps(serializable_globals, indent=4)


def hash_str(input: str) -> str:
    return hashlib.sha256(input.encode()).hexdigest()[:32]


def hash_str_to_int(input: str) -> int:
    return int(hashlib.sha256(input.encode()).hexdigest(), base=16)


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


client_connections: dict[asyncio.AbstractEventLoop, httpx.AsyncClient] = {}


def get_client() -> httpx.AsyncClient:
    event_loop = asyncio.get_event_loop()
    if event_loop not in client_connections:
        client_connections[event_loop] = httpx.AsyncClient()
    return client_connections[event_loop]


async def wrap_sem[T](f: Awaitable[T], sem: asyncio.Semaphore) -> T:
    async with sem:
        return await f


def unzip[A, B](pairs: list[tuple[A, B]]) -> tuple[list[A], list[B]]:
    return tuple(map(list, zip(*pairs, strict=True))) if len(pairs) > 0 else ([], [])  # pyright: ignore[reportReturnType]


def unzip3[A, B, C](pairs: list[tuple[A, B, C]]) -> tuple[list[A], list[B], list[C]]:
    return (
        tuple(map(list, zip(*pairs, strict=True))) if len(pairs) > 0 else ([], [], [])
    )  # pyright: ignore[reportReturnType]


def unzip4[A, B, C, D](
    pairs: list[tuple[A, B, C, D]],
) -> tuple[list[A], list[B], list[C], list[D]]:
    return (
        tuple(map(list, zip(*pairs, strict=True)))
        if len(pairs) > 0
        else ([], [], [], [])
    )  # pyright: ignore[reportReturnType]


async def async_zip[T, *Ts](f: Awaitable[T], *args: *Ts) -> tuple[T, *Ts]:
    result = await f
    return (result, *args)


def avg(values: list[float]) -> float:
    if len(values) == 0:
        return float("nan")
    else:
        return sum(values) / len(values)


def argsort(values: list[float]) -> list[int]:
    return sorted(range(len(values)), key=lambda i: values[i])


def sorted_by_keys[T](
    values: list[T], keys: list[float], *, reverse: bool = False
) -> list[T]:
    """Sort values by corresponding scores in ascending order."""
    return [
        value
        for value, _ in sorted(
            zip(values, keys, strict=True), key=lambda x: x[1], reverse=reverse
        )
    ]


def clamp(value: float, minimum: float, maximum: float) -> float:
    assert minimum <= maximum
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value


def flatten[T](array_2d: list[list[T]]) -> list[T]:
    return [item for array in array_2d for item in array]


def unwrap[T](value: T | None) -> T:
    assert value is not None
    return value


def read_num_lines_pbar(file_path: str, *, display_name: str | None = None) -> int:
    if display_name is None:
        display_name = file_path
    num_lines = 0
    with open(file_path) as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        f.seek(0)
        with tqdm(
            total=file_size,
            unit="B",
            unit_scale=True,
            desc=f"Reading {display_name}",
        ) as pbar:
            for line in f:
                num_lines += 1
                pbar.set_postfix(
                    {
                        "Lines": str(num_lines),
                    }
                )
                pbar.update(len(line.encode("utf-8")))
    return num_lines


@contextmanager
def lock_file(f: IO[Any]) -> Generator[None, Any, None]:
    try:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield None
    finally:
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)


def load_custom_dataset(path: str) -> Dataset | DatasetDict:
    dataset_dict_path = f"{path}/dataset_dict.json"
    if os.path.exists(dataset_dict_path):
        with open(dataset_dict_path) as f:
            dataset_dict_splits = [
                str(elem) for elem in list(json.loads(f.read())["splits"])
            ]
            return DatasetDict(
                {
                    split_name: load_custom_dataset(f"{path}/{split_name}")
                    for split_name in dataset_dict_splits
                }
            )
    else:
        return load_from_disk(path)
