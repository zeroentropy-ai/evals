from evals.ai import AIRerankModel, ai_rerank
from zeroentropy_eval import rerank_batch
import numpy as np
import json
import os
import aiohttp
import asyncio
from tqdm import tqdm

async def process_query(model, query, corpus_texts):
    if model == "zeroentropy":
        client = aiohttp.ClientSession()
        reranked_scores = await rerank_batch(
            [[query, corpus_text] for corpus_text in corpus_texts],
            client
        )
        await client.close()
    else:
        reranked_scores = await ai_rerank(
            model,
            query,
            corpus_texts,
        )
    return reranked_scores.tolist() if isinstance(reranked_scores, np.ndarray) else reranked_scores

def load_embedding_data(dataset_name):
    """Load and parse data from embeddings.json"""
    data_file = f"results/{dataset_name}/embedding_data.json"
    with open(data_file, "r") as f:
        data = json.load(f)
    
    # Convert lists back to numpy arrays
    dot_products = np.array(data["dot_products"])
    top_100_indices = np.array(data["top_100_indices"])
    
    # Reconstruct corpus and queries
    corpus = {
        "text": data["corpus_texts"],
        "_id": data["corpus_ids"]
    }
    queries = {
        "text": data["query_texts"],
        "_id": data["query_ids"]
    }
    
    return {
        "dot_products": dot_products,
        "top_100_indices": top_100_indices,
        "corpus": corpus,
        "queries": queries
    }

async def main():
    dataset_name = "clinia/CUREv1"
    company = "cohere"  # or "together" or "zeroentropy"
    model_name = "rerank-v3.5" #"Salesforce/Llama-Rank-V1"  or "zeroentropy"
    
    # Load relevance scores from standardized format
    print("Loading relevance scores...")
    with open("default.json", "r") as f:
        relations = json.load(f)
    
    # Load and parse embedding data
    print("Loading embedding data...")
    data = load_embedding_data(dataset_name)
    dot_products = data["dot_products"]
    top_100_indices = data["top_100_indices"]
    corpus = data["corpus"]
    queries = data["queries"]
    
    # Create mappings
    query_indices = {queries["_id"][i]: i for i in range(len(queries["_id"]))}
    doc_indices = {corpus["_id"][i]: i for i in range(len(corpus["_id"]))}
    
    # Initialize relevance scores
    relevance_scores = np.zeros((len(queries["_id"]), 100), dtype=np.float32)
    
    # Fill relevance scores
    print("Computing relevance scores...")
    for i in tqdm(range(len(relations["query-id"]))):
        query_id = relations["query-id"][i]
        doc_id = relations["corpus-id"][i]
        score = relations["score"][i]
        
        if query_id in query_indices:
            current_arr = top_100_indices[query_indices[query_id]].tolist() if isinstance(top_100_indices[query_indices[query_id]], np.ndarray) else top_100_indices[query_indices[query_id]]
            doc_index = current_arr.index(doc_indices[doc_id]) if doc_indices[doc_id] in current_arr else 99
            relevance_scores[query_indices[query_id], doc_index] = score
    
    print(f"Running reranker: {company}/{model_name}")
    
    # Initialize model
    if company != 'zeroentropy':
        model = AIRerankModel(company=company, model=model_name)
    else:
        model = "zeroentropy"
    
    try:
        # Truncate texts to avoid token limits
        short_corpus = [text[:16384] for text in corpus["text"]]
        short_queries = [text[:16384] for text in queries["text"]]
        
        # Create tasks for all queries
        tasks = []
        for i in range(len(queries["_id"])):
            # Get the top 100 documents for this query
            query_docs = [short_corpus[j] for j in top_100_indices[i]]
            task = process_query(
                model,
                short_queries[i],
                query_docs
            )
            tasks.append(task)
        
        # Process queries in batches
        batch_size = 10
        if model_name == "Salesforce/Llama-Rank-V1":
            model_name = "Llama-Rank-V1"  # no / for file saving please!
        
        # Setup results file
        results_dir = f"results/{dataset_name}"
        os.makedirs(results_dir, exist_ok=True)
        results_file = f"{results_dir}/{company}_{model_name}.jsonl"
        
        # Clear file if it exists
        with open(results_file, "w") as f:
            f.truncate(0)
        
        # Process batches and save results
        with open(results_file, "a") as f:
            for i in tqdm(range(0, len(tasks), batch_size), desc="Processing batches"):
                batch = tasks[i:i + batch_size]
                results = await asyncio.gather(*batch)
                
                for j, reranked_scores in enumerate(results):
                    query_idx = i + j
                    desired_scores = relevance_scores[query_idx].tolist()
                    dot_scores = dot_products[query_idx][top_100_indices[query_idx]].tolist()
                    
                    f.write(json.dumps({
                        "reranked_scores": reranked_scores,
                        "query_id": queries["_id"][query_idx],
                        "relevance_scores": desired_scores,
                        "dot_product_scores": dot_scores,
                        "top_100_doc_ids": [corpus["_id"][idx] for idx in top_100_indices[query_idx]]
                    }))
                    f.write("\n")
        
        print(f"Results saved to {results_file}")
        
    except Exception as e:
        print(f"Error processing queries for {company}/{model_name}: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 