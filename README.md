# ZeroEntropy AI Evaluation Suite

This repository contains ZeroEntropy's comprehensive evaluation suite for benchmarking and testing AI models across various retrieval and reranking tasks. The system supports multiple datasets, embedding methods, and reranking models with sophisticated metrics calculation.

## Overview

The evaluation pipeline consists of four main stages:
1. **Data Ingestion** - Load and preprocess datasets from various sources
2. **Embedding Generation** - Create vector embeddings using different retrieval methods
3. **Reranking** - Apply reranking models to improve retrieval results
4. **Metrics Calculation** - Compute NDCG, Recall, and other evaluation metrics

## Installation

### Prerequisites
- uv
- CUDA-capable GPU (optional, for local model inference)

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd evals

# Install dependencies using uv (recommended) or pip
uv sync

# Set up environment variables
cp .env.example .env # Edit .env with your API keys and configuration
```

### Required Environment Variables
```bash
# ZeroEntropy
ZEROENTROPY_API_KEY=your_zeroentropy_key
# OpenAI
OPENAI_API_KEY=your_openai_key
# Anthropic (optional)
ANTHROPIC_API_KEY=your_anthropic_key
# Cohere (optional)
COHERE_API_KEY=your_cohere_key
# VoyageAI (optional)
VOYAGEAI_API_KEY=your_voyage_key
# Jina AI (optional)
JINA_API_KEY=your_jina_key
# Modal (optional, for custom models)
MODAL_KEY=your_modal_key
MODAL_SECRET=your_modal_secret
# Baseten (optional)
BASETEN_API_KEY=your_baseten_key
# Together AI (optional)
TOGETHER_API_KEY=your_together_key
```

## Quick Start

### Running the Complete Pipeline
```bash
# Run the full evaluation pipeline (all stages)
python evals/run_pipeline.py
```

### Running Individual Components

#### 1. Data Ingestion
```bash
# Ingest all default datasets
python evals/run_ingestors.py

# Ingest with custom parameters
python -c "
from evals.run_ingestors import run_ingestors
from evals.types import ALL_INGESTORS
run_ingestors(ingestors=ALL_INGESTORS[:5], max_queries=50)
"
```

#### 2. Generate Embeddings
```bash
# Generate embeddings with default settings
python evals/run_embeddings.py

# Generate embeddings with custom retrieval method
python -c "
import asyncio
from evals.run_embeddings import run_embeddings
asyncio.run(run_embeddings(retrieval_method='bm25'))
"
```

#### 3. Run Rerankers
```bash
# Run reranking with default models
python evals/run_rerankers.py

# Run with specific rerankers
python -c "
import asyncio
from evals.run_rerankers import run_rerankers
asyncio.run(run_rerankers(rerankers=['cohere', 'zeroentropy-large']))
"
```

#### 4. Calculate Metrics
```bash
# Calculate NDCG metrics
python evals/run_ndcg.py

# Calculate Recall metrics
python evals/run_recall.py
```

## Configuration

### Supported Retrieval Methods
- `openai_small` - OpenAI text-embedding-3-small (default)
- `qwen3_4b` - Qwen3 4B embedding model
- `qwen3_0.6b` - Qwen3 0.6B embedding model
- `bm25` - BM25 keyword-based retrieval
- `hybrid` - Combination of embedding and BM25 methods

### Supported Rerankers
- **ZeroEntropy Models**: `zeroentropy-large`, `zeroentropy-small`
- **Commercial APIs**: `cohere`, `jina`, `voyageai`
- **Open Source**: `mixedbread`, `qwen`, `salesforce`
- **Embedding-based**: `openai-large-embedding`

### Available Datasets

#### Original Datasets
- **FiQA** - Financial question answering
- **BioASQ** - Biomedical questions
- **StackOverflow QA** - Programming questions
- **MS MARCO** - Web search queries
- **Financial Benchmarks** - FinQABench, FinanceBench
- **Code Datasets** - MBPP, CosQA
- **Legal Datasets** - Various legal document retrieval tasks

#### MTEB Datasets
- **Multilingual**: TwitterHjerneRetrieval, MLQARetrieval, WikipediaRetrievalMultilingual
- **English**: ArguAna, SCIDOCS, TRECCOVID, WinoGrande
- **Specialized**: LEMBPasskeyRetrieval, TempReasonL1, SpartQA

#### New Datasets
- **Programming**: LeetCode (Python, Java, JavaScript, C++)
- **Q&A**: Quora, Quora Swedish
- **Documents**: Meeting transcripts, NarrativeQA, Pandas documentation

## Advanced Usage

### Custom Configuration
```python
from evals.run_pipeline import run_pipeline
from evals.types import MTEB_INGESTORS
import asyncio

# Run pipeline with MTEB datasets only
INGESTORS = MTEB_INGESTORS[:5]  # First 5 MTEB datasets
RETRIEVAL_METHOD = "hybrid"
RERANKERS = ["zeroentropy-large", "cohere"]

async def custom_run():
    await run_pipeline("ingestors", "ndcg")

asyncio.run(custom_run())
```

### Adding Custom Datasets
```python
from evals.ingestors.common import BaseIngestor
from evals.common import Document, Query, QRel

class CustomIngestor(BaseIngestor):
    def dataset_id(self) -> str:
        return "custom/my_dataset"

    def ingest(self) -> tuple[list[Query], list[Document], list[QRel]]:
        # Load your data
        queries = [Query(id="q1", query="What is AI?")]
        documents = [Document(id="d1", content="AI is artificial intelligence")]
        qrels = [QRel(query_id="q1", document_id="d1", score=1.0)]
        return queries, documents, qrels
```

### Running on Train Split

To run evaluations on the training split of datasets instead of test/validation, you need to modify the dataset configuration. Many ingestors have a `split` parameter:

```python
from evals.ingestors.master_mteb_ingestion import MasterMtebIngestor

# Example: Run on train split
train_ingestor = MasterMtebIngestor(
    task_name="TwitterHjerneRetrieval",
    dataset_name="twitterhjerneretrieval",
    language="dan-Latn",
    split="train"  # Change this to "train"
)

# Use in pipeline
from evals.run_ingestors import run_ingestors
run_ingestors(ingestors=[train_ingestor])
```

For custom datasets, modify the split parameter in the ingestor configuration in `evals/types.py`.

## Output Structure

Results are stored in `{ROOT}/data/datasets/` with the following structure:
```
data/datasets/
├── {dataset_id}/
│   ├── queries.jsonl           # Processed queries
│   ├── documents.jsonl         # Processed documents
│   ├── qrels.jsonl            # Relevance judgments
│   └── {retrieval_method}/
│       ├── ze_results.jsonl        # Initial retrieval results
│       ├── embeddings_cache.db     # Embedding cache
│       └── {reranker}/
│           └── ze_scores.jsonl  # Reranked results
```

## Performance Tips

1. **Caching**: Embeddings are cached automatically to speed up reruns
2. **Parallel Processing**: Reranking supports concurrent processing
3. **Memory Management**: Large datasets are processed in batches
4. **Rate Limiting**: Built-in rate limiting for all API providers
5. **GPU Usage**: Local models automatically use CUDA if available

## Metrics

### NDCG (Normalized Discounted Cumulative Gain)
Measures ranking quality with position-based discounting:
```bash
python evals/run_ndcg.py
```

### Recall@K
Measures the fraction of relevant documents retrieved in top K:
```bash
python evals/run_recall.py
```

Both metrics support:
- Per-dataset breakdowns
- Statistical significance testing (standard error calculation)
- Comparison across retrieval methods and rerankers

## Development

### Project Structure
```
evals/
├── ai.py                   # AI model interfaces and utilities
├── common.py              # Core data types and configurations
├── types.py               # Type definitions and defaults
├── utils.py               # Utility functions
├── run_*.py              # Main execution scripts
└── ingestors/            # Dataset-specific ingestion logic
    ├── common.py         # Base ingestor class
    ├── fiqa.py          # Example dataset ingestor
    └── ...              # Other dataset ingestors
```

### Adding New Models

#### Embedding Models
Add to `ALL_RERANKERS` in `evals/types.py`:
```python
"my-embedding": AIEmbeddingModel(company="my_company", model="my-model"),
```

#### Reranking Models
Add to `ALL_RERANKERS` in `evals/types.py`:
```python
"my-reranker": AIRerankModel(company="my_company", model="my-model"),
```

### Testing
```bash
# Run linting
./lint.sh

# Test individual components
python evals/run_ingestors.py
python -c "import asyncio; from evals.run_embeddings import run_embeddings; asyncio.run(run_embeddings())"
```

## Troubleshooting

### Common Issues

1. **API Rate Limits**: The system includes automatic rate limiting, but you may need to adjust limits in `evals/ai.py`

2. **Memory Issues**: Reduce batch sizes or dataset sizes:
   ```python
   run_ingestors(max_queries=50)  # Limit to 50 queries per dataset
   ```

3. **Missing Dependencies**: Ensure all optional dependencies are installed:
   ```bash
   pip install torch sentence-transformers rank_bm25
   ```

4. **GPU Issues**: Set device explicitly:
   ```python
   # In evals/ai.py, modify DEVICE variable
   DEVICE = "cpu"  # Force CPU usage
   ```

### Getting Help

- Check the logs in `{ROOT}/logs/*`
- Review the configuration in `evals/types.py`
- Examine individual ingestor implementations in `evals/ingestors/*`
- Ask us on [Slack](https://go.zeroentropy.dev/slack) or [Discord](https://go.zeroentropy.dev/discord)

## License

This project is developed by the ZeroEntropy AI team. See the license file for usage terms.
