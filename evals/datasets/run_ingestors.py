import os

from evals.datasets.ingestors.bioasq import BioasqIngestor
from evals.datasets.ingestors.common import BaseIngestor, clean_dataset, limit_queries
from evals.datasets.ingestors.cosqa import CosqaIngestor
from evals.datasets.ingestors.curev1 import CureV1Ingestor
from evals.datasets.ingestors.financebench import FinancebenchIngestor
from evals.datasets.ingestors.finqabench import FinqabenchIngestor
from evals.datasets.ingestors.fiqa import FiqaIngestor
from evals.datasets.ingestors.ineqs import IneqsIngestor
from evals.datasets.ingestors.master_legal_ingestion import MasterLegalIngestor
from evals.datasets.ingestors.mbpp import MbppIngestor
from evals.datasets.ingestors.msmarco import MSMarcoIngestor
from evals.datasets.ingestors.qmsum import QMSumIngestor
from evals.datasets.ingestors.stackoverflowqa import StackoverflowqaIngestor

# from evals.datasets.ingestion.narrativeqa import NarrativeQAIngestor
# from evals.datasets.ingestion.meeting import MeetingIngestor
from evals.utils import ROOT

MAX_QUERIES = 1000

BASE_PATH = f"{ROOT}/data/datasets"
INGESTORS: list[BaseIngestor] = [
    FiqaIngestor(),
    BioasqIngestor(),
    StackoverflowqaIngestor(),
    QMSumIngestor(),
    MSMarcoIngestor(),
    FinqabenchIngestor(),
    FinancebenchIngestor(),
    MbppIngestor(),
    CureV1Ingestor(),
    IneqsIngestor(),
    CosqaIngestor(),
    MasterLegalIngestor(
        "evals/legalquad",
        dataset_name="mteb/LegalQuAD",
        language="german",
        split="test",
    ),
    MasterLegalIngestor(
        "evals/lecardv2",
        dataset_name="mteb/LeCaRDv2",
        language="chinese",
        split="test",
    ),
    MasterLegalIngestor(
        "evals/legalsum",
        dataset_name="mteb/legal_summarization",
        language="english",
        split="test",
    ),
    MasterLegalIngestor(
        "evals/aila",
        dataset_name="mteb/AILA_casedocs",
        language="english",
        split="test",
    ),
    MasterLegalIngestor(
        "evals/consumercontracts",
        dataset_name="mteb/legalbench_consumer_contracts_qa",
        language="english",
        split="test",
    ),
    MasterLegalIngestor(
        "evals/corporatelobbying",
        dataset_name="mteb/legalbench_corporate_lobbying",
        language="english",
        split="test",
    ),
    # MasterLegalIngestor(
    #     "evals/gerdalir",
    #     dataset_name="mteb/GerDaLIRSmall",
    #     language="german",
    #     split="test",
    # ),
    #MeetingIngestor(),
    #NarrativeQAIngestor(),
]


EVAL_DATASETS = [ingestor.dataset() for ingestor in INGESTORS]

if __name__ == "__main__":
    for i, ingestor in enumerate(INGESTORS):
        dataset = ingestor.dataset()
        print(f"===> Ingesting {dataset.id} (Dataset {i + 1}/{len(INGESTORS)}) <===")

        # Create dataset directory if it doesn't exist
        dataset_dir = dataset.root_path
        os.makedirs(dataset_dir, exist_ok=True)

        # Run Ingestion
        queries, documents, qrels = clean_dataset(*ingestor.ingest())
        queries, documents, qrels = limit_queries(
            queries,
            documents,
            qrels,
            limit=MAX_QUERIES,
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
