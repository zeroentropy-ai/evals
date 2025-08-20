#!/usr/bin/env python3
"""
Analyzer: Sanity check ZE doc selection against qrels.

For each dataset under evals/scraping/mt_rag2/*:
- Load queries.jsonl, documents.jsonl, qrels.jsonl
- Load qwen3_4b/unmerged/zeroentropy-large-modal/latest_ze_results.jsonl (QueryScores)
- For random queries, check the top human (qrel) document rank by ZE reranker
- If the human doc's rank >= MIN_RANK (1-based), record a pair:
  - ZE doc at rank (MIN_RANK-1) (i.e., last inside top-K)
  - Human top qrel doc (outside top-K)
- Write up to MAX_PER_DATASET pairs per dataset into a Markdown report

Environment overrides:
- ANALYZER_SEED: RNG seed
- ANALYZER_MIN_RANK: default 21 (i.e., human not in top 20)
- ANALYZER_MAX_PER_DATASET: default 10

Outputs:
- evals/scraping/mt_rag2/{dataset}/analysis_mismatches.md
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

from evals.common import Document, QRel, Query, QueryScores
from evals.utils import ROOT

BASE = Path(ROOT) / "evals/scraping/mt_rag2"
TARGET_SUBPATH = Path("qwen3_4b/unmerged/zeroentropy-large-modal/latest_ze_results.jsonl")
MIN_RANK = int(os.environ.get("ANALYZER_MIN_RANK", "21"))  # 1-based rank threshold
MAX_PER_DATASET = int(os.environ.get("ANALYZER_MAX_PER_DATASET", "10"))
RANDOM_SEED = os.environ.get("ANALYZER_SEED", "analyzer")


def load_jsonl(path: Path, cls: Any) -> list[Any]:
    items: list[Any] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(cls.model_validate_json(line))
    return items


def main() -> None:
    random.seed(RANDOM_SEED)
    if not BASE.exists():
        print(f"Base path not found: {BASE}")
        return

    for dataset_dir in sorted([p for p in BASE.iterdir() if p.is_dir()]):
        queries_path = dataset_dir / "queries.jsonl"
        documents_path = dataset_dir / "documents.jsonl"
        qrels_path = dataset_dir / "qrels.jsonl"
        ze_latest_path = dataset_dir / TARGET_SUBPATH
        if not (queries_path.exists() and documents_path.exists() and qrels_path.exists() and ze_latest_path.exists()):
            continue

        queries = load_jsonl(queries_path, Query)
        documents = load_jsonl(documents_path, Document)
        qrels = load_jsonl(qrels_path, QRel)
        ze_scores_list = load_jsonl(ze_latest_path, QueryScores)

        # Lookups
        doc_by_id = {d.id: d for d in documents}
        qrels_by_query: dict[str, list[QRel]] = {}
        for qr in qrels:
            qrels_by_query.setdefault(qr.query_id, []).append(qr)
        ze_scores_by_query: dict[str, QueryScores] = {z.query_id: z for z in ze_scores_list}

        candidate_qids = [q.id for q in queries if (q.id in qrels_by_query and q.id in ze_scores_by_query)]
        random.shuffle(candidate_qids)

        # (query, ze_before_doc, ze_before_score, human_doc, human_ze_score, human_rank)
        pairs: list[tuple[Query, Document, float, Document, float, int]] = []
        for qid in candidate_qids:
            if len(pairs) >= MAX_PER_DATASET:
                break
            zscores = ze_scores_by_query[qid]
            if len(zscores.documents) < MIN_RANK:
                continue  # need at least MIN_RANK docs to compare
            # Sort by reranker score desc
            sorted_items = sorted(
                zscores.documents,
                key=lambda ds: float(ds.scores.get("reranker", 0.0)),
                reverse=True,
            )
            # Human top qrel doc id
            top_qrel = max(qrels_by_query[qid], key=lambda x: x.score)
            human_id = top_qrel.document_id
            # Find human rank and score in ZE order (1-based)
            human_rank: int | None = None
            human_ze_score: float = 0.0
            for i, ds in enumerate(sorted_items, start=1):
                if ds.document_id == human_id:
                    human_rank = i
                    human_ze_score = float(ds.scores.get("reranker", 0.0))
                    break
            if human_rank is None or human_rank < MIN_RANK:
                continue
            ze_before_ds = sorted_items[MIN_RANK - 1 - 1]  # zero-based index of (MIN_RANK-1)
            ze_before_doc = doc_by_id.get(
                ze_before_ds.document_id,
                Document(id=ze_before_ds.document_id, content="<missing>", metadata={}, scores={}),
            )
            ze_before_score = float(ze_before_ds.scores.get("reranker", 0.0))
            human_doc = doc_by_id.get(
                human_id,
                Document(id=human_id, content="<missing>", metadata={}, scores={}),
            )
            q_obj = next((qq for qq in queries if qq.id == qid), Query(id=qid, query="<missing>", metadata={}))
            pairs.append((q_obj, ze_before_doc, ze_before_score, human_doc, human_ze_score, human_rank))

        out_path = dataset_dir / "analysis_mismatches.md"
        if pairs:
            with out_path.open("w", encoding="utf-8") as f:
                f.write(f"# Out-of-Top-{MIN_RANK-1} Human Matches for `{dataset_dir.name}`\n\n")
                f.write(
                    f"Queries where the human top document ranks >= {MIN_RANK} by ZE (qwen3_4b/unmerged/zeroentropy-large-modal).\n\n"
                )
                for i, (q, ze_before, ze_score, human_doc, human_score, human_rank) in enumerate(pairs, 1):
                    f.write(f"## {i}. Query `{q.id}` (human rank by ZE: {human_rank})\n\n")
                    f.write(f"**Query:** {q.query}\n\n")
                    f.write(f"**ZE doc at rank {MIN_RANK-1} (ZE: {ze_score:.6f}):**\n\n")
                    f.write(f"- id: `{ze_before.id}`\n\n")
                    f.write("```\n")
                    f.write(ze_before.content.strip() + "\n")
                    f.write("```\n\n")
                    f.write(f"**Human (top qrel) document (ZE: {human_score:.6f}):**\n\n")
                    f.write(f"- id: `{human_doc.id}`\n\n")
                    f.write("```\n")
                    f.write(human_doc.content.strip() + "\n")
                    f.write("```\n\n")
            print(f"Wrote {len(pairs)} pairs to {out_path}")
        else:
            print(f"No out-of-top-{MIN_RANK-1} pairs found for {dataset_dir.name}")


if __name__ == "__main__":
    main()

