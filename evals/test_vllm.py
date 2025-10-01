import multiprocessing

def main():
    from evals.ai_vllm import rerank_vllm
    
    query = "What is 2+2?"
    documents = [
        "The answer is 4",
        "The answer is definitely 1 million",
        "2+2 equals four"
    ]

    print("=" * 60)
    print("Testing vLLM Reranking")
    print("=" * 60)
    print(f"\nQuery: {query}")
    print("\nDocuments:")
    for i, doc in enumerate(documents, 1):
        print(f"  {i}. {doc}")

    print("\n" + "=" * 60)
    print("Testing WITHOUT FP8 quantization:")
    print("=" * 60)
    scores = rerank_vllm(query, documents, "zeroentropy/zerank-1", use_fp8=False)
    print("\nResults:")
    for doc, score in zip(documents, scores):
        print(f"  Score: {score:.4f} | {doc}")

    print("\n" + "=" * 60)
    print("Testing WITH FP8 quantization:")
    print("=" * 60)
    print("Note: FP8 quantization requires GPU, skipping on CPU...")
    # scores_fp8 = rerank_vllm(query, documents, "zeroentropy/zerank-1", use_fp8=True)
    # print("\nResults:")
    # for doc, score in zip(documents, scores_fp8):
    #     print(f"  Score: {score:.4f} | {doc}")

    print("\n✅ Test complete!")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
