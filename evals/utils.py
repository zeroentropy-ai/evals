import asyncio
import hashlib
import os
from pathlib import Path
from collections.abc import Awaitable
import aiohttp
from tqdm import tqdm

ROOT = f"{Path(__file__).resolve().parent.parent}"

def hash_str(input: str) -> str:
    return hashlib.sha256(input.encode()).hexdigest()[:32]

def avg(values: list[float]) -> float:
    if len(values) == 0:
        return float("nan")
    else:
        return sum(values) / len(values)


def argsort(values: list[float]) -> list[int]:
    return sorted(range(len(values)), key=lambda i: values[i])

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

async def wrap_sem[T](f: Awaitable[T], sem: asyncio.Semaphore) -> T:
    async with sem:
        return await f

client_connections: dict[asyncio.AbstractEventLoop, aiohttp.ClientSession] = {}

def get_client() -> aiohttp.ClientSession:
    event_loop = asyncio.get_event_loop()
    if event_loop not in client_connections:
        client_connections[event_loop] = aiohttp.ClientSession()
    return client_connections[event_loop]