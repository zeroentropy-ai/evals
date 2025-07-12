import os

from evals.ingestors.common import BaseIngestor, clean_dataset, limit_queries
from evals.types import DEFAULT_INGESTORS, DEFAULT_MAX_QUERIES


def main(
    *,
    ingestors: list[BaseIngestor] = DEFAULT_INGESTORS,
    max_queries: int = DEFAULT_MAX_QUERIES,
) -> None:
    for i, ingestor in enumerate(ingestors):
        dataset = ingestor.dataset()
        print(f"===> Ingesting {dataset.id} (Dataset {i + 1}/{len(ingestors)}) <===")

        # Create dataset directory if it doesn't exist
        dataset_dir = dataset.root_path
        os.makedirs(dataset_dir, exist_ok=True)

        # Run Ingestion
        queries, documents, qrels = clean_dataset(*ingestor.ingest())
        queries, documents, qrels = limit_queries(
            queries,
            documents,
            qrels,
            limit=max_queries,
            seed=dataset.id,
        )

        # Write the results
        with open(f"{dataset_dir}/queries.jsonl", "w") as f:
            for q in queries:
                f.write(q.model_dump_json() + "\n")
        with open(f"{dataset_dir}/documents.jsonl", "w") as f:
            for d in documents:
                f.write(d.model_dump_json() + "\n")
        with open(f"{dataset_dir}/qrels.jsonl", "w") as f:
            for qrel in qrels:
                f.write(qrel.model_dump_json() + "\n")


if __name__ == "__main__":
    main()
