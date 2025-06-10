from datasets import load_dataset
from evals.ai import AIRerankModel, ai_rerank, AIEmbeddingModel, ai_embedding, AIEmbeddingType
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
    top_hits = load_dataset("mteb/LeCaRDv2", "default")
    queries = load_dataset("mteb/LeCaRDv2", "queries")
    corpus = load_dataset("mteb/LeCaRDv2", "corpus")
    top_hits = top_hits["test"]
    queries = queries["queries"]
    corpus = corpus["corpus"]
    print("Document Corpus:", corpus)
    print("Queries:", queries)
    print("Relevance Scores Dataset:", top_hits)
    query_indices = {queries["_id"][i]: i for i in range(queries.num_rows)}
    doc_indices = {corpus["_id"][i]: i for i in range(corpus.num_rows)}
    relevance_scores = np.zeros((queries.num_rows, min(100, corpus.num_rows)), dtype=np.float32)
    is_small = corpus.num_rows <= 100

    if is_small:
        for i in range(top_hits.num_rows):
            query_id = top_hits["query-id"][i]
            doc_id = top_hits["corpus-id"][i]
            score = top_hits["score"][i]
            if query_id in query_indices and doc_id in doc_indices:
                relevance_scores[query_indices[query_id], doc_indices[doc_id]] = score

    else: #add an embeddings step to our pipeline
        print("Too many documents, doing embeddings to filter out top 100")
        embedding_model = AIEmbeddingModel(company="openai", model="text-embedding-3-small")
        most_similar_documents = {queries["_id"][i]: [] for i in range(queries.num_rows)}

        #gotta batch this later. Careful!
        document_embeddings = np.array(await ai_embedding(model=embedding_model, 
                                                          texts = corpus["text"], 
                                                          embedding_type=AIEmbeddingType.DOCUMENT))
        query_embeddings = np.array(await ai_embedding(model=embedding_model, 
                                                       texts = queries["text"], 
                                                       embedding_type=AIEmbeddingType.QUERY))

        print("Embeddings generated, doing similarity search now.")        
        for i in range(queries.num_rows):
            print(i)
            current_query_embedding = query_embeddings[i]
            all_cosine_similarities = (document_embeddings @ current_query_embedding.T).T
            top_100_doc_indices = (np.argsort(all_cosine_similarities)[-100:])[::-1]
            most_similar_documents[queries["_id"][i]] = list(map(lambda index: corpus["_id"][index], top_100_doc_indices))
        
        print("Similarity search complete. Setting up reranking")
        for i in range(top_hits.num_rows):
            print(i)
            query_id = top_hits["query-id"][i]
            doc_id = top_hits["corpus-id"][i]
            score = top_hits["score"][i]
            if query_id in query_indices and doc_id in most_similar_documents[query_id]:
                relevance_scores[query_indices[query_id], most_similar_documents[query_id].index(doc_id)] = score
            else:
                #this means the desired doc is ranked lower than 100th place. Let's just make it the 100th place doc.
                relevance_scores[query_indices[query_id], 99] = score
                most_similar_documents[query_id][-1] = doc_id



    model_companies = ["cohere", "together", "zeroentropy"]
    model_names = ["rerank-v3.5", "Salesforce/Llama-Rank-V1", "zeroentropy"]
    
    # Evaluate each model
    for company, model_name in zip(model_companies, model_names):
        print(f"Evaluating company {company} with model {model_name}")
        
        if company != 'zeroentropy':
            model = AIRerankModel(company=company, model=model_name)
        else:
            model = "zeroentropy"
        
        # Create tasks for all queries
        print(sum([len(text) for text in corpus["text"]])/len(corpus["text"]))
        short_corpus = [corpus["text"][j][:20000] for j in range(len(corpus["text"]))]
        tasks = []
        for i in range(queries.num_rows):
            if is_small:
                task = process_query(
                    model,
                    queries["text"][i],
                    short_corpus,
                    relevance_scores[i]
                )
                tasks.append(task)
            else:
                task = process_query(
                    model,
                    queries["text"][i],
                    [short_corpus[j] for j in most_similar_documents[queries["_id"][i]]],
                    relevance_scores[i]
                )
                tasks.append(task)
        
        # Process queries in batches of 50 for progress tracking
        batch_size = 10# if company == "together" else 50
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