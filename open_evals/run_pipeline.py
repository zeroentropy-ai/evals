from open_evals.run_ingestors import main as run_ingestors
from open_evals.run_embeddings import main as run_embeddings
from open_evals.run_rerankers import main as run_rerankers
from open_evals.run_ndcg import main as run_ndcg
import asyncio

def main() -> None:
    run_ingestors()
    asyncio.run(run_embeddings())
    asyncio.run(run_rerankers())
    run_ndcg()

if __name__ == "__main__":
    main()