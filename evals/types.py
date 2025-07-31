from evals.ai import AIEmbeddingModel, AIRerankModel
from evals.common import RerankerName, RetrievalMethod
from evals.ingestors.bioasq import BioasqIngestor
from evals.ingestors.common import BaseIngestor
from evals.ingestors.cosqa import CosqaIngestor
from evals.ingestors.curev1 import CureV1Ingestor
from evals.ingestors.financebench import FinancebenchIngestor
from evals.ingestors.finqabench import FinqabenchIngestor
from evals.ingestors.fiqa import FiqaIngestor
from evals.ingestors.ineqs import IneqsIngestor
from evals.ingestors.leetcode_multi import LeetcodeMultiLanguageIngestor
from evals.ingestors.master_legal_ingestion import MasterLegalIngestor
from evals.ingestors.master_mteb_ingestion import MasterMtebIngestor
from evals.ingestors.mbpp import MbppIngestor
from evals.ingestors.meeting import MeetingIngestor
from evals.ingestors.msmarco import MSMarcoIngestor
from evals.ingestors.narrativeqa import NarrativeQAIngestor
from evals.ingestors.pandas import PandasIngestor
from evals.ingestors.qmsum import QMSumIngestor
from evals.ingestors.quora import QuoraIngestor
from evals.ingestors.quora_swedish import QuoraSwedishIngestor
from evals.ingestors.stackoverflowqa import StackoverflowqaIngestor

ALL_RERANKERS: dict[RerankerName, AIRerankModel | AIEmbeddingModel] = {
    "cohere": AIRerankModel(company="cohere", model="rerank-v3.5"),
    "salesforce": AIRerankModel(company="together", model="Salesforce/Llama-Rank-V1"),
    "zeroentropy-large": AIRerankModel(company="zeroentropy", model="zerank-1"),
    "zeroentropy-small": AIRerankModel(company="zeroentropy", model="zerank-1-small"),
    "zeroentropy-large-modal": AIRerankModel(
        company="modal",
        model="https://zeroentropy--ze-rerank-v0-3-0-dev-model-endpoint.modal.run/",
    ),
    "zeroentropy-small-modal": AIRerankModel(
        company="modal",
        model="https://zeroentropy--ze-rerank-small-v0-3-0-dev-model-endpoint.modal.run/",
    ),
    "zeroentropy-baseten": AIRerankModel(
        company="baseten",
        model="https://model-4w5l09vq.api.baseten.co/environments/production/async_predict",
    ),
    "mixbread": AIRerankModel(
        company="huggingface", model="mixedbread-ai/mxbai-rerank-large-v1"
    ),
    "jina": AIRerankModel(company="jina", model="jina-reranker-m0"),
    "qwen": AIRerankModel(
        company="modal",
        model="https://zeroentropy--qwen3-reranker-4b-model-endpoint.modal.run/",
    ),
    "openai-large-embedding": AIEmbeddingModel(
        company="openai",
        model="text-embedding-3-large",
    ),
}

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
        instructions="Given a claim, find documents that refute the claim:",
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
    # ),
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
ALL_INGESTORS = MTEB_INGESTORS + NEW_INGESTORS + OLD_INGESTORS
DEFAULT_INGESTORS = MTEB_INGESTORS + NEW_INGESTORS + OLD_INGESTORS
DEFAULT_MAX_QUERIES = 1000
DEFAULT_RETRIEVAL_METHOD: RetrievalMethod = "openai_small"
DEFAULT_INCLUDE_RELEVANT_DOCS: bool = True
DEFAULT_RERANKERS: list[RerankerName] = [
    "zeroentropy-small",
    "zeroentropy-large",
    "zeroentropy-baseten",
]
