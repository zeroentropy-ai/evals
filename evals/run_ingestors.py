import os

from pydantic import BaseModel

from evals.ingestors.common import BaseIngestor, limit_queries, validate_dataset
from evals.types import DEFAULT_INGESTORS, DEFAULT_MAX_QUERIES


def write_jsonl[T: BaseModel](file_name: str, models: list[T]) -> None:
    existing_lines: list[str] | None = None
    if os.path.exists(file_name):
        existing_lines = []
        with open(file_name) as f:
            for line in f:
                existing_lines.append(line.strip())

    output_lines: list[str] = []
    for model in models:
        model_string = model.model_dump_json()
        output_lines.append(model_string.strip())

    if output_lines == existing_lines:
        print(f"=> {file_name} is identical. The file was not modified.")
    else:
        with open(file_name, "w") as f:
            for line in output_lines:
                f.write(line + "\n")
        print(f"=> Wrote to {file_name}")


def run_ingestors(
    *,
    ingestors: list[BaseIngestor] = DEFAULT_INGESTORS,
    max_queries: int = DEFAULT_MAX_QUERIES,
) -> None:
    for i, ingestor in enumerate(ingestors):
        dataset = ingestor.dataset()
        print(f"===> Ingesting {dataset.id} (Dataset {i + 1}/{len(ingestors)}) <===")

        # Create dataset directory if it doesn't exist
        os.makedirs(dataset.root_path, exist_ok=True)

        # Run Ingestion
        queries, documents, qrels = ingestor.ingest()
        queries, documents, qrels = validate_dataset(queries, documents, qrels)
        queries, documents, qrels = limit_queries(
            queries,
            documents,
            qrels,
            limit=max_queries,
            seed=dataset.id,
        )

        # Write the results
        write_jsonl(dataset.queries_path, queries)
        write_jsonl(dataset.documents_path, documents)
        write_jsonl(dataset.qrels_path, qrels)


if __name__ == "__main__":
    run_ingestors()
