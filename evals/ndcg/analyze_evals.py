import json
import numpy as np

def analyze_ndcg(company, model_name, dataset_name):
    results_path = f"results/{dataset_name}/{company}_{model_name}.jsonl"
    ndcgs = []
    try:
        with open(results_path, "r") as f:
            for line in f:
                result = json.loads(line)
                reranked_scores = np.array(result["reranked_scores"])
                relevance_scores = np.array(result["relevance_scores"])

                # Compute DCG for the reranked order
                reranked_order = np.argsort(reranked_scores)[::-1]
                #if model_name == "zeroentropy": #a peculiarity
                #    reranked_order = np.argsort(reranked_scores)
                #else:
                #    reranked_order = np.argsort(reranked_scores)[::-1]
                dcg = sum(
                    relevance_scores[idx] / np.log2(rank + 2)
                    for rank, idx in enumerate(reranked_order)
                )

                # Compute IDCG (ideal DCG)
                ideal_order = np.argsort(relevance_scores)[::-1]
                idcg = sum(
                    relevance_scores[idx] / np.log2(rank + 2)
                    for rank, idx in enumerate(ideal_order)
                )

                ndcg = dcg / idcg if idcg > 0 else None
                if ndcg is not None:
                    ndcgs.append(ndcg)

        average_ndcg = np.mean(ndcgs) if ndcgs else 0.0
        print(f"Average NDCG for {company}/{model_name}: {average_ndcg}")
        return average_ndcg
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    dataset_name = "clinia/CUREv1"
    analyze_ndcg("cohere", "rerank-v3.5", dataset_name)
    analyze_ndcg("together", "Llama-Rank-V1", dataset_name)
    analyze_ndcg("zeroentropy", "zeroentropy", dataset_name)