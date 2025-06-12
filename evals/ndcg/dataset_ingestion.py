from datasets import load_dataset

def ingest(dataset_name, format):

    if format == "qdr": #the standard nice format
        queries = load_dataset(dataset_name, "queries")
        corpus = load_dataset(dataset_name, "corpus")
        relations = load_dataset(dataset_name, "default")
        queries.to_json("queries.json")
        corpus.to_json("corpus.json")
        relations.to_json("default.json")

    elif format == "posexamples": #another semi-common format, needs some subcustomization
        dataset = load_dataset(dataset_name)
        dataset = dataset["train"]
        queries = {}
        queries["text"] = dataset["prompt"]
        queries["_id"] = [i for i in range(len(queries["text"]))]
        corpus = {}
        corpus["text"] = dataset["test_list"][0]
        for i in range(1, len(dataset["test_list"])):
            corpus["text"].extend(dataset["test_list"][i])
        corpus["_id"] = [i for i in range(len(corpus["text"]))]
        relations = {}
        relations["query-id"] = []
        relations["corpus-id"] = []
        relations["score"] = []
        document_id = 0
        for i in range(len(dataset["prompt"])):
            for test in dataset["test_list"][i]:
                relations["query-id"].append(i)
                relations["corpus-id"].append(document_id)
                relations["score"].append(1)
                document_id += 1
        import json
        with open("queries.json", "w") as f:
            json.dump(queries, f)
        with open("corpus.json", "w") as f:
            json.dump(corpus, f)
        with open("default.json", "w") as f:
            json.dump(relations, f)
            
    elif format == "susformat": #custom code required here.
        #in this case, it is QDR but within the QDR there are a ton of annoying sub-splits that need to be annealed.
        queries = load_dataset(dataset_name, "queries-en")
        corpus = load_dataset(dataset_name, "corpus")
        relations = load_dataset(dataset_name, "qrels")
        # Concatenate all sub-datasets in `queries` column-wise into a single dataset
        from datasets import concatenate_datasets

        # queries is a DatasetDict with keys for each sub-dataset
        all_query_datasets = [queries[key] for key in queries.keys()]
        merged_queries = concatenate_datasets(all_query_datasets)
        all_corpus_datasets = [corpus[key] for key in corpus.keys()]
        merged_corpus = concatenate_datasets(all_corpus_datasets)
        all_relations_datasets = [relations[key] for key in relations.keys()]
        merged_relations = concatenate_datasets(all_relations_datasets)
        print(merged_queries)
        merged_queries.to_json("queries.json")
        merged_corpus.to_json("corpus.json")
        merged_relations.to_json("default.json")
    else:
        raise ValueError(f"Invalid format: {format}")

if __name__ == "__main__":
    ingest("clinia/CUREv1", "susformat")