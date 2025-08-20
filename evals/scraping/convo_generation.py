#!/usr/bin/env python3
"""
Build an Airbnb conversation dataset from ./data/airbnb using OpenAI to generate
multi-turn back-and-forth conversations (user <-> assistant) that culminate in
one final question answerable by the initial document.

Run:
  python -m evals.scraping.convo_generation \
    --limit-unique 1000 \
    --model gpt-4o-mini \
    --max-concurrency 32 \
    --max-retries 5 \
    --temperature 1.4
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
from collections.abc import Awaitable, Sequence
from pathlib import Path
from typing import TypedDict, cast

from dotenv import load_dotenv
from tqdm import tqdm

from evals.ai import (
    AIMessage,
    AIModel,
    ai_call,
    tiktoken_truncate_by_num_tokens,
)
from evals.common import Document, QRel, Query
from evals.utils import ROOT

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(ROOT) / "data/airbnb"
DATASET_DIR = Path(ROOT) / "data/datasets/airbnb_convo"
QUERIES_PATH = DATASET_DIR / "queries.jsonl"
DOCUMENTS_PATH = DATASET_DIR / "documents.jsonl"
QRELS_PATH = DATASET_DIR / "qrels.jsonl"

# Default constants (do not reassign these)
LIMIT_UNIQUE_DEFAULT = 1000
MODEL_DEFAULT = "gpt-4o-mini"
MAX_CONCURRENCY_DEFAULT = 32
MAX_RETRIES_DEFAULT = 5
TEMPERATURE_DEFAULT = 1.4

# Token limits for truncation attempts (progressively stricter)
DOC_TOKEN_LIMITS = [8000, 4000, 2000, 1000, 500]


def ensure_dirs() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)


def list_markdown_files(directory: Path) -> list[Path]:
    def sort_key(p: Path) -> tuple[int, str]:
        try:
            return (0, f"{int(p.stem):010d}")
        except ValueError:
            return (1, p.stem)
    return sorted([p for p in directory.iterdir() if p.is_file() and p.suffix == ".md"], key=sort_key)


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logger.warning(f"Failed to read {path}: {e}")
        return ""


def extract_core_markdown(text: str) -> str:
    lines = text.splitlines()
    start = 0
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("markdown content:"):
            start = i + 1
            break
    body = "\n".join(lines[start:])
    body = "\n".join(
        l for l in body.splitlines()
        if not l.strip().lower().startswith("title:") and not l.strip().lower().startswith("url source:")
    )
    body = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", body)
    body = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", body)
    body = body.lower()
    body = re.sub(r"\s+", " ", body).strip()
    return body


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def write_jsonl(path: Path, models: Sequence[Document | Query | QRel]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for m in models:
            f.write(m.model_dump_json() + "\n")


class ConvoJSON(TypedDict, total=False):
    conversation: list[dict[str, str]]
    final_question: str


def make_prompt(idx: str, doc_text: str) -> list[AIMessage]:
    system = (
        "You are a creative assistant that writes realistic, quirky human conversations that build towards a final question, "
        "which must be answerable using ONLY the provided document."
    )
    user = (
        "Create a short back-and-forth conversation (3-5 turns, alternating user and assistant).\n"
        "- Each turn should be natural and may reference prior turns.\n"
        "- The final user message must be a single concise question that is uniquely answerable from the document.\n"
        "- Do NOT include quotes or preambles.\n"
        "- Conversations can be weird/human-like, but keep the final question coherent and grounded in the doc.\n"
        "Return JSON with keys: conversation (list of {role: 'user'|'assistant', content: str}), final_question: str.\n\n"
        f"Document (id={idx}):\n\n{doc_text}\n"
    )
    return [
        AIMessage(role="system", content=system),
        AIMessage(role="user", content=user),
    ]


def try_parse_json(text: str) -> ConvoJSON | None:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return cast(ConvoJSON, cast(object, obj))
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return cast(ConvoJSON, cast(object, obj))
        except Exception:
            return None
    return None


async def generate_one_convo(
    model_name: str,
    idx: str,
    doc_text: str,
    *,
    max_retries: int,
    temperature: float,
) -> tuple[str, str, list[dict[str, str]]]:
    model = AIModel(company="openai", model=model_name)
    temp = max(0.0, min(float(temperature), 2.0))

    for attempt in range(1, max_retries + 1):
        limit_idx = min(attempt - 1, len(DOC_TOKEN_LIMITS) - 1)
        max_tokens_for_doc = DOC_TOKEN_LIMITS[limit_idx]
        pruned_doc = tiktoken_truncate_by_num_tokens(doc_text, max_tokens_for_doc)
        messages = make_prompt(idx, pruned_doc)
        try:
            response_text = await ai_call(
                model,
                messages,
                temperature=temp,
            )
            data = try_parse_json(response_text)
            if data is None:
                raise ValueError("Model did not return valid JSON")
            conv_input = data.get("conversation", [])
            final_question = data.get("final_question", "").strip()
            conv_items: list[dict[str, str]] = []
            for item in conv_input:
                role_value = item.get("role", "")
                content_value = item.get("content", "")
                if role_value in ("user", "assistant") and content_value:
                    conv_items.append({"role": role_value, "content": content_value})
            if len(final_question) == 0:
                raise ValueError("Empty final_question")
            return idx, final_question, conv_items
        except Exception as e:
            if attempt == max_retries:
                logger.warning(f"[{idx}] failed after {attempt} attempts: {e}")
                return idx, "", []
        await asyncio.sleep(min(5.0, 0.5 * (2 ** (attempt - 1))))
    return idx, "", []


async def generate_all_convos(
    docs: list[Document],
    *,
    model_name: str,
    max_concurrency: int,
    max_retries: int,
    temperature: float,
) -> list[tuple[str, str, list[dict[str, str]]]]:
    load_dotenv(override=True)
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Please add it to your .env")

    sem = asyncio.Semaphore(max_concurrency)

    async def bound(d: Document) -> tuple[str, str, list[dict[str, str]]]:
        async with sem:
            return await generate_one_convo(
                model_name,
                d.id,
                d.content,
                max_retries=max_retries,
                temperature=temperature,
            )

    tasks: list[Awaitable[tuple[str, str, list[dict[str, str]]]]] = [bound(d) for d in docs]

    results: list[tuple[str, str, list[dict[str, str]]]] = []
    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating Conversations"):
        results.append(await fut)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Airbnb conversation dataset")
    parser.add_argument("--limit-unique", type=int, default=LIMIT_UNIQUE_DEFAULT)
    parser.add_argument("--model", type=str, default=MODEL_DEFAULT)
    parser.add_argument("--max-concurrency", type=int, default=MAX_CONCURRENCY_DEFAULT)
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES_DEFAULT)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE_DEFAULT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    files = list_markdown_files(DATA_DIR)
    if len(files) == 0:
        raise FileNotFoundError(f"No markdown files found under {DATA_DIR}")

    seen_hashes: set[str] = set()
    documents: list[Document] = []
    for path in files:
        raw = read_text(path)
        if not raw.strip():
            continue
        core = extract_core_markdown(raw)
        if not core:
            continue
        key = sha256(core)
        if key in seen_hashes:
            continue
        seen_hashes.add(key)
        idx = str(len(documents))
        documents.append(Document(id=idx, content=raw, metadata={}))

    if len(documents) == 0:
        raise RuntimeError("No unique documents processed; nothing to write")

    write_jsonl(DOCUMENTS_PATH, documents)
    logger.info(f"Wrote documents: {len(documents)} -> {DOCUMENTS_PATH}")

    docs_for_convo = documents[: args.limit_unique]
    convos = asyncio.run(
        generate_all_convos(
            docs_for_convo,
            model_name=args.model,
            max_concurrency=args.max_concurrency,
            max_retries=args.max_retries,
            temperature=args.temperature,
        )
    )
    id_to_convo: dict[str, tuple[str, list[dict[str, str]]]] = {idx: (q, conv) for idx, q, conv in convos}

    queries: list[Query] = []
    qrels: list[QRel] = []
    missing = 0
    for d in docs_for_convo:
        final_q, conv_items = id_to_convo.get(d.id, ("", []))
        final_q = final_q.strip()
        if len(final_q) == 0:
            missing += 1
            final_q = "What is the primary issue discussed in this Airbnb Help article?"
        queries.append(Query(id=d.id, query=final_q, metadata={"conversation": conv_items}))
        qrels.append(QRel(query_id=d.id, document_id=d.id, score=1.0))

    if missing > 0:
        logger.warning(f"Filled {missing} missing final questions with fallback prompts.")

    write_jsonl(QUERIES_PATH, queries)
    write_jsonl(QRELS_PATH, qrels)

    logger.info(
        f"Wrote dataset to {DATASET_DIR} (queries={len(queries)}, documents={len(documents)}, qrels={len(qrels)})"
    )


if __name__ == "__main__":
    main()
