from datasets import load_dataset
from evals.ai import AIRerankModel, ai_rerank
from zeroentropy_eval import rerank_batch
import numpy as np
import asyncio
import aiohttp

async def process_query(model, query, corpus_texts, relevance_score_row):
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
    idcg = sum(
        relevance_score_row[ind] / np.log2(idx + 2) for idx, ind in enumerate(
            np.argsort(relevance_score_row)[::-1]
        )
    )
    dcg = sum([relevance_score_row[ind] / np.log2(idx + 2) for idx, ind in enumerate(np.argsort(reranked_scores)[::-1])])
    ndcg = dcg / idcg if idcg > 0 else None
    return ndcg

async def main():
    top_hits = load_dataset("mteb/legalbench_consumer_contracts_qa", "default")
    queries = load_dataset("mteb/legalbench_consumer_contracts_qa", "queries")
    corpus = load_dataset("mteb/legalbench_consumer_contracts_qa", "corpus")
    top_hits = top_hits["test"]
    queries = queries["queries"]
    corpus = corpus["corpus"]
    print("Document Corpus:", corpus)
    print("Queries:", queries)
    print("Relevance Scores Dataset:", top_hits)
    query_indices = {queries["_id"][i]: i for i in range(queries.num_rows)}
    doc_indices = {corpus["_id"][i]: i for i in range(corpus.num_rows)}
    relevance_scores = np.zeros((queries.num_rows, corpus.num_rows), dtype=np.float32)
    for i in range(top_hits.num_rows):
        query_id = top_hits["query-id"][i]
        doc_id = top_hits["corpus-id"][i]
        score = top_hits["score"][i]
        if query_id in query_indices and doc_id in doc_indices:
            relevance_scores[query_indices[query_id], doc_indices[doc_id]] = score

    model_companies = ["cohere", "together", "zeroentropy"]
    model_names = ["rerank-v3.5", "Salesforce/Llama-Rank-V1", "zeroentropy"]
    
    # Evaluate each model
    for company, model_name in zip(model_companies, model_names):
        print(f"Evaluating company {company} with model {model_name}")
        
        if company != 'zeroentropy':
            model = AIRerankModel(company=company, model=model_name)
        else:
            model = "zeroentropy"
        
        if company == "together":
            #these guys need a smaller context size. 20000 chars should be less that 8192 tokens.
            short_corpus = [corpus["text"][j][:20000] for j in range(len(corpus["text"]))]
            tasks = []
            for i in range(queries.num_rows):
                task = process_query(
                    model,
                    queries["text"][i][:20000],
                    short_corpus,
                    relevance_scores[i]
                )
                tasks.append(task)
        else:
        # Create tasks for all queries
            tasks = []
            for i in range(queries.num_rows):
                task = process_query(
                    model,
                    queries["text"][i],
                    corpus["text"],
                    relevance_scores[i]
                )
                tasks.append(task)
        
        # Process queries in batches of 50 for progress tracking
        batch_size = 10
        total_ndcg = 0.0
        total_valid = 0
        
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            results = await asyncio.gather(*batch)
            
            for ndcg in results:
                if ndcg is not None:
                    total_ndcg += ndcg
                    total_valid += 1
            
            print(f"Processed {min(i + batch_size, len(tasks))} queries")

        average_ndcg = total_ndcg / total_valid if total_valid > 0 else 0.0
        print(f"Average NDCG for company {company}/{model_name}: {average_ndcg}")


if __name__ == "__main__":
    asyncio.run(main())