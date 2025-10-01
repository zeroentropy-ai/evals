import asyncio
from evals.ai import ai_rerank, AIRerankModel

async def test_rerankers():
    query = "What is 2+2?"
    documents = [
        "The answer is 4",
        "The answer is definitely 1 million",
        "2+2 equals four"
    ]
    
    print("=" * 60)
    print("Testing Reranker Comparison")
    print("=" * 60)
    print(f"\nQuery: {query}")
    print("\nDocuments:")
    for i, doc in enumerate(documents, 1):
        print(f"  {i}. {doc}")
    
    # Test ZeroEntropy API (this will work without GPU)
    print("\n" + "=" * 60)
    print("Testing ZeroEntropy API (zerank-1):")
    print("=" * 60)
    
    ze_model = AIRerankModel(company="zeroentropy", model="zerank-1")
    scores_api = await ai_rerank(ze_model, query, documents)
    
    print("\nResults:")
    for doc, score in zip(documents, scores_api):
        print(f"  Score: {score:.4f} | {doc}")
    
    print("\n✅ API test complete!")
    print("\nNote: To test vLLM FP8 quantization, you need to run this on a machine with a GPU.")

if __name__ == "__main__":
    asyncio.run(test_rerankers())
