import httpx
from typing import List
import asyncio

FASTAPI_URL = "http://localhost:8002"
MAX_BATCH_SIZE = 100  # Process 100 documents at a time

async def rerank_fastapi(
    query: str,
    documents: List[str],
) -> List[float]:
    """Rerank documents using the FastAPI server with batching."""
    
    # If small batch, send directly
    if len(documents) <= MAX_BATCH_SIZE:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{FASTAPI_URL}/rerank",
                json={
                    "query": query,
                    "documents": documents,
                }
            )
            response.raise_for_status()
            result = response.json()
            return result["scores"]
    
    # For large batches, split and process in parallel
    all_scores = []
    tasks = []
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        for i in range(0, len(documents), MAX_BATCH_SIZE):
            batch = documents[i:i + MAX_BATCH_SIZE]
            
            async def process_batch(batch_docs):
                response = await client.post(
                    f"{FASTAPI_URL}/rerank",
                    json={
                        "query": query,
                        "documents": batch_docs,
                    }
                )
                response.raise_for_status()
                result = response.json()
                return result["scores"]
            
            tasks.append(process_batch(batch))
        
        # Process all batches in parallel
        results = await asyncio.gather(*tasks)
        
        # Flatten results
        for batch_scores in results:
            all_scores.extend(batch_scores)
    
    return all_scores
