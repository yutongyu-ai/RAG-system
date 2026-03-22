import os
import json
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder

client = OpenAI()  # Initialize OpenAI API client for embeddings

reranker = CrossEncoder("BAAI/bge-reranker-base")  # Load reranker
bge_base_model = SentenceTransformer("BAAI/bge-base-en")  # Load BGE-base model
bge_large_model = SentenceTransformer("BAAI/bge-large-en")  # Load BGE-large model

def rerank(query, docs, top_k=5):
    """
    Rerank retrieved documents using a cross-encoder and return the top-k most relevant ones
    """
    pairs = [[query, d["text"]] for d in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [x[0] for x in ranked[:top_k]]

def cosine_similarity(a, b):
    """
    Compute the cosine similarity between two vectors.
    """
    score = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    return score

def embed_query(query, model_name="openai"):
    """
    Generate the embedding vector for the query using the specified model.
    """
    if model_name == "openai":
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        return response.data[0].embedding

    elif model_name == "bge-base":
        query = "Represent this sentence for retrieval: " + query
        return bge_base_model.encode([query], normalize_embeddings=True)[0]

    elif model_name == "bge-large":
        query = "Represent this sentence for retrieval: " + query
        return bge_large_model.encode([query], normalize_embeddings=True)[0]

    else:
        raise ValueError(f"Unknown model: {model_name}")

def retrieve(query, vector_data, model_name="openai", top_k=5, use_rerank=True):
    """
    Retrieve top-k relevant documents using embedding similarity.
    """
    query_emb = embed_query(query, model_name)

    scores = []

    for item in vector_data:
        score = cosine_similarity(query_emb, item["embedding"])

        scores.append({
            "id": item["id"],
            "score": score,
            "text": item["text"],
            "metadata": item.get("metadata", {})
        })

    scores = sorted(scores, key=lambda x: x["score"], reverse=True)

    if use_rerank:
        candidates = scores[:20]
        return rerank(query, candidates, top_k=top_k)

    return scores[:top_k]

if __name__ == "__main__":
    with open("vector_store/vector_store_bge_base.json", "r", encoding="utf-8") as f:
        vector_data = json.load(f)

    with open("benchmark.json", "r", encoding="utf-8") as f:
        queries = json.load(f)

    all_outputs = []

    for q in queries:
        query = q["question"]

        # Retrieve top-k results (with optional reranking)
        results = retrieve(query, vector_data, model_name="bge-base", top_k=5, use_rerank=True)

        formatted_results = []

        for i, r in enumerate(results, start=1):
            metadata = r.get("metadata", {})

            # Format each retrieved result with ranking and metadata
            formatted_results.append({
                "rank": i,
                "score": r["score"],
                "retrieved_answer": r["text"],
                "retrieved_source": metadata.get("source", ""),
                "retrieved_parent_doc_id": metadata.get("parent_doc_id", ""),
                "retrieved_chunk_id": r["id"]
            })
        # Format the final output for the query
        output_item = {
            "id": q["id"],
            "question": q["question"],
            "question_type": q.get("question_type", ""),
            "difficulty": q.get("difficulty", ""),

            "retrieved": formatted_results,

            # Ground truth information for evaluation
            "gold_answer": q.get("gold_answer", ""),
            "gold_parent_doc_id": q.get("gold_parent_doc_id", []),
            "gold_chunk_ids": q.get("gold_chunk_ids", [])
        }

        all_outputs.append(output_item)

    folder_path = "retrieval_results"
    file_path = os.path.join(folder_path, "retrieved_bge_base_rerank_results.json")
    os.makedirs(folder_path, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(all_outputs, f, ensure_ascii=False, indent=2)

    print("All retrieval results saved to retrieval_results")

