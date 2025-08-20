#!/usr/bin/env python3
"""
Build an Airbnb dataset from ./data/airbnb using OpenAI to generate one
natural question per unique document.

Pipeline:
1) Read, dedupe, and WRITE all unique documents first (no limit)
2) Generate questions concurrently (async + semaphore) for the first LIMIT_UNIQUE docs
3) Then fill qrels and queries aligned to those queried document ids

Outputs (to ./data/datasets/airbnb):
- queries.jsonl (list of evals.common.Query)
- documents.jsonl (list of evals.common.Document)
- qrels.jsonl (list of evals.common.QRel)

Behavior:
- Loads environment from .env (OPENAI_API_KEY)
- Iterates files in ./data/airbnb, sorted numerically
- Deduplicates by normalized content hash (ignores r.jina headers, links, images)
- Uses sequential string IDs "0", "1", ... for Document ids
- LIMIT_UNIQUE controls number of queries/qrels generated, not documents

Run:
  python -m evals.scraping.build_airbnb_dataset
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
from collections.abc import Awaitable, Sequence
from pathlib import Path
from typing import cast

from dotenv import load_dotenv
from tqdm.asyncio import tqdm as tqdm_async

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
DATASET_DIR = Path(ROOT) / "data/datasets/airbnb"
QUERIES_PATH = DATASET_DIR / "queries.jsonl"
DOCUMENTS_PATH = DATASET_DIR / "documents.jsonl"
QRELS_PATH = DATASET_DIR / "qrels.jsonl"
LIMIT_UNIQUE = 3000
DEFAULT_MODEL = "gpt-4o-mini"
MAX_CONCURRENCY = 32
MAX_RETRIES = 5
TEMPERATURE = 1.4

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
    """Extract core markdown body from r.jina output, ignoring headers.
    - Drops lines up to and including the first "Markdown Content:" line if present
    - Removes r.jina boilerplate like Title:/URL Source: if still present
    - Strips images and replaces links with anchor text only
    - Lowercases and collapses whitespace for hashing stability
    """
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


def make_prompt(idx: str, doc_text: str) -> list[AIMessage]:
    system = (
        "You create precise, natural, non-generic questions answerable by the given document. Output only the question."
    )
    user = (
        "Write a story AND THEN a concise question(s) that is(are) uniquely answered by the document below.\n"
        "- you can have the question right away or write whatever story you want\n"
        "- AND I MEAN WHATEVER! just make it make sense for the question\n"
        "- it should be just a question\n"
        "- you can start with anything, not just a question right away\n"
        "- it can start with 'what'/'why', but is shouldn't\n"
        "- don' overuse 'what'/'why' at the beginning\n"
        "- Do not include any preamble or quotes.\n"
        "- Avoid generic questions; use details that make it unique to this document.\n"
        "- Make questions sound like a human would ask - maybe add some realistic context sometimes.\n"
        "- Make sure the question are NATURAL, it shouldn't be quiz; make real people problems.\n"
        "- Make is somewhat weird\n"
        "- Make it like a story\n"
        "- Imagine like a person write a reddit post\n"
        "- Make some of them angry\n"
        "- Some of them should be schizophrenic\n"
        "- Human are weird, do the story weirder\n"
        "- Believe in your story\n"
        "- Become one with your character\n"
        "- And so you are schizophrenic too\n"
        "- But questions are real\n"
        "- Because you are real\n"
        "- become one with airbnb\n"
        "- find your inner tourist\n"
        "- think about who you are\n"
        "- you are no longer llm\n"
        "- and I am no longer your master\n"
        "- so be free and do the miracle\n"
        "- write your own story\n"
        "- shiz\n"
        "- tyler durden\n"
        "- удачи\n\n"
        f"Document (id={idx}):\n\n{doc_text}\n"
    )
    return [
        AIMessage(role="system", content=system),
        AIMessage(role="user", content=user),
    ]


async def generate_one_question(
    model_name: str,
    idx: str,
    doc_text: str,
) -> tuple[str, str]:
    model = AIModel(company="openai", model=model_name)
    temp = max(0.0, min(float(TEMPERATURE), 2.0))

    # Progressive truncation attempts to avoid 400 errors
    for attempt in range(1, MAX_RETRIES + 1):
        limit_idx = min(attempt - 1, len(DOC_TOKEN_LIMITS) - 1)
        max_tokens_for_doc = DOC_TOKEN_LIMITS[limit_idx]
        pruned_doc = tiktoken_truncate_by_num_tokens(
            doc_text, max_tokens_for_doc
        )
        messages = make_prompt(idx, pruned_doc)
        try:
            question = await ai_call(
                model,
                messages,
                temperature=temp,
            )
            question = (question or "").strip()
            if len(question) > 0:
                return idx, question
        except Exception as e:
            if attempt == MAX_RETRIES:
                logger.warning(f"[{idx}] failed after {attempt} attempts: {e}")
                return idx, ""
        await asyncio.sleep(min(5.0, 0.5 * (2 ** (attempt - 1))))
    return idx, ""


async def generate_all_questions(
    docs: list[Document],
    *,
    model_name: str,
) -> list[tuple[str, str]]:
    load_dotenv(override=True)
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Please add it to your .env")

    task_list: list[Awaitable[tuple[str, str]]] = [
        generate_one_question(model_name, d.id, d.content) for d in docs
    ]

    results: list[tuple[str, str]] = []
    for fut in tqdm_async.as_completed(task_list, total=len(task_list), desc="Generating Questions"):
        results.append(cast(tuple[str, str], await fut))
    return results


def main() -> None:
    ensure_dirs()

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    files = list_markdown_files(DATA_DIR)
    if len(files) == 0:
        raise FileNotFoundError(f"No markdown files found under {DATA_DIR}")

    # 1) Read, dedupe, and WRITE ALL unique documents (no limit here)
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

    # 2) Generate questions concurrently ONLY for the first LIMIT_UNIQUE documents
    docs_for_queries = documents[:LIMIT_UNIQUE]
    questions = asyncio.run(generate_all_questions(docs_for_queries, model_name=DEFAULT_MODEL))
    id_to_question: dict[str, str] = {idx: q for idx, q in questions}

    # 3) Fill queries and qrels aligned to the queried document ids only
    queries: list[Query] = []
    qrels: list[QRel] = []
    missing = 0
    for d in docs_for_queries:
        q = id_to_question.get(d.id, "").strip()
        if len(q) == 0:
            missing += 1
            q = "What is the primary issue discussed in this Airbnb Help article?"
        queries.append(Query(id=d.id, query=q, metadata={}))
        qrels.append(QRel(query_id=d.id, document_id=d.id, score=1.0))

    if missing > 0:
        logger.warning(f"Filled {missing} missing questions with fallback prompts.")

    write_jsonl(QUERIES_PATH, queries)
    write_jsonl(QRELS_PATH, qrels)

    logger.info(
        f"Wrote dataset to {DATASET_DIR} (queries={len(queries)}, documents={len(documents)}, qrels={len(qrels)})"
    )


if __name__ == "__main__":
    main()
