from evals.ai import AIEmbeddingModel, ai_embedding, AIEmbeddingType
import numpy as np
import json
import os
from tqdm import tqdm

async def main():
    dataset_name = "clinia/CUREv1"
    
    # Load standardized dataset format
    print("Loading dataset...")
    with open("queries.json", "r") as f:
        queries = json.load(f)
    with open("corpus.json", "r") as f:
        corpus = json.load(f)
    
    # Handle large query sets
    if len(queries["text"]) > 1000:
        random_indices = np.random.choice(len(queries["text"]), 1000, replace=False)
        queries = {
            "_id": [queries["_id"][i] for i in random_indices],
            "text": [queries["text"][i] for i in random_indices]
        }

    print("Generating embeddings...")
    embedding_model = AIEmbeddingModel(company="openai", model="text-embedding-3-small")
    
    # Generate embeddings
    document_embeddings = np.array(await ai_embedding(
        model=embedding_model,
        texts=corpus["text"],
        embedding_type=AIEmbeddingType.DOCUMENT
    ))
    query_embeddings = np.array(await ai_embedding(
        model=embedding_model,
        texts=queries["text"],
        embedding_type=AIEmbeddingType.QUERY
    ))

    # Calculate dot products
    print("Computing dot products...")
    dot_products = query_embeddings @ document_embeddings.T  # (num_queries, num_docs)
    
    # Get top 100 documents for each query
    print("Finding top 100 documents...")
    top_100_indices = np.argpartition(dot_products, -100, axis=1)[:, -100:]  # (num_queries, 100)
    sorted_top_100_indices = np.argsort(
        np.take_along_axis(dot_products, top_100_indices, axis=1), axis=1
    )[:, ::-1]  # (num_queries, 100)
    top_100_sorted_indices = np.take_along_axis(top_100_indices, sorted_top_100_indices, axis=1)
    
    # Create results directory
    results_dir = f"results/{dataset_name}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save all necessary data
    data_file = f"{results_dir}/embedding_data.json"
    with open(data_file, "w") as f:
        json.dump({
            "dot_products": dot_products.tolist(),
            "top_100_indices": top_100_sorted_indices.tolist(),
            "query_ids": queries["_id"],
            "query_texts": queries["text"],
            "corpus_ids": corpus["_id"],
            "corpus_texts": corpus["text"]
        }, f)
    
    print(f"Data saved to {data_file}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 