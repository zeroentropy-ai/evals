import os

from open_evals.ingestors.bioasq import BioasqIngestor
from open_evals.ingestors.common import BaseIngestor, clean_dataset, limit_queries
from open_evals.ingestors.cosqa import CosqaIngestor
from open_evals.ingestors.curev1 import CureV1Ingestor
from open_evals.ingestors.financebench import FinancebenchIngestor
from open_evals.ingestors.finqabench import FinqabenchIngestor
from open_evals.ingestors.fiqa import FiqaIngestor
from open_evals.ingestors.ineqs import IneqsIngestor
from open_evals.ingestors.master_legal_ingestion import MasterLegalIngestor
from open_evals.ingestors.mbpp import MbppIngestor
from open_evals.ingestors.msmarco import MSMarcoIngestor
from open_evals.ingestors.qmsum import QMSumIngestor
from open_evals.ingestors.stackoverflowqa import StackoverflowqaIngestor

from open_evals.ingestors.leetcode_multi import LeetcodeMultiLanguageIngestor
from open_evals.ingestors.narrativeqa import NarrativeQAIngestor
from open_evals.ingestors.meeting import MeetingIngestor
from open_evals.ingestors.quora import QuoraIngestor
from open_evals.ingestors.quora_swedish import QuoraSwedishIngestor
from open_evals.ingestors.pandas import PandasIngestor
from open_evals.ingestors.master_mteb_ingestion import MasterMtebIngestor
from open_evals.utils import ROOT

MAX_QUERIES = 1000

BASE_PATH = f"{ROOT}/data/datasets"
OLD_INGESTORS: list[BaseIngestor] = [
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
]
NEW_INGESTORS: list[BaseIngestor] = [
    LeetcodeMultiLanguageIngestor(language="python"),
    LeetcodeMultiLanguageIngestor(language="java"),
    LeetcodeMultiLanguageIngestor(language="javascript"),
    LeetcodeMultiLanguageIngestor(language="c++"),
    QuoraIngestor(),
    QuoraSwedishIngestor(),
    MeetingIngestor(),
    NarrativeQAIngestor(),
    PandasIngestor(),
]

MTEB_INGESTORS: list[BaseIngestor] = [
    MasterMtebIngestor(
        task_name="TwitterHjerneRetrieval",
        dataset_name="twitterhjerneretrieval",
        language="dan-Latn",
        split="train",
    ),
    MasterMtebIngestor(
        task_name="ArguAna",
        dataset_name="arguana",
        language="eng-Latn",
        split="test",
    ),
    MasterMtebIngestor(
        task_name="HagridRetrieval",
        dataset_name="hagridretrieval",
        language="eng-Latn",
        split="dev",
    ),
    MasterMtebIngestor(
        task_name="LEMBPasskeyRetrieval",
        dataset_name="lembpasskeyretrieval",
        language="eng-Latn",
        split="test_256",
    ),
    MasterMtebIngestor(
        task_name="SCIDOCS",
        dataset_name="scidocs",
        language="eng-Latn",
        split="test",
    ),
    MasterMtebIngestor(
        task_name="SpartQA",
        dataset_name="spartqa",
        language="eng-Latn",
        split="test",
    ),
    MasterMtebIngestor(
        task_name="TempReasonL1",
        dataset_name="tempreasonl1",
        language="eng-Latn",
        split="test",
    ),
    MasterMtebIngestor(
        task_name="TRECCOVID",
        dataset_name="treccovid",
        language="eng-Latn",
        split="test",
    ),
    MasterMtebIngestor(
        task_name="WinoGrande",
        dataset_name="winogrande",
        language="eng-Latn",
        split="test",
    ),
    # MasterMtebIngestor(
    #    task_name="BelebeleRetrieval",
    #    dataset_name="belebeleretrieval",
    #    language="multilingual",
    #    split="test",
    #),
    MasterMtebIngestor(
        task_name="MLQARetrieval",
        dataset_name="mlqaretrieval",
        language="multilingual",
        split="test",
    ),
    MasterMtebIngestor(
        task_name="StatcanDialogueDatasetRetrieval",
        dataset_name="statcandialoguedatasetretrieval",
        language="multilingual",
        split="test",
    ),
    MasterMtebIngestor(
        task_name="WikipediaRetrievalMultilingual",
        dataset_name="wikipediaretrievalmultilingual",
        language="multilingual",
        split="test",
    ),
    MasterMtebIngestor(
        task_name="CovidRetrieval",
        dataset_name="covidretrieval",
        language="cmn-Hans",
        split="dev",
    ),
    MasterMtebIngestor(
        task_name="MIRACLRetrievalHardNegatives",
        dataset_name="miraclretrievalhardnegatives",
        language="multilingual",
        split="test",
    ),
]

INGESTORS = MTEB_INGESTORS + NEW_INGESTORS + OLD_INGESTORS

EVAL_DATASETS = [ingestor.dataset() for ingestor in INGESTORS]

def main() -> None:
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

if __name__ == "__main__":
    main()
