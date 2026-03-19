import json
import numpy as np
from openai import OpenAI

client = OpenAI()

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def embed_query(query):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    return response.data[0].embedding

def retrieve(query, vector_data, top_k=3):
    query_emb = embed_query(query)

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

    return scores[:top_k]

if __name__ == "__main__":
    with open("vector_store.json", "r", encoding="utf-8") as f:
        vector_data = json.load(f)

    with open("benchmark.json", "r", encoding="utf-8") as f:
        queries = json.load(f)

    all_outputs = []

    for q in queries:
        query = q["question"]

        results = retrieve(query, vector_data)

        formatted_results = []

        for i, r in enumerate(results, start=1):
            metadata = r.get("metadata", {})

            formatted_results.append({
                "rank": i,
                "score": r["score"],
                "retrieved_answer": r["text"],
                "retrieved_source": metadata.get("source", ""),
                "retrieved_parent_doc_id": metadata.get("parent_doc_id", ""),
                "retrieved_chunk_id": r["id"]
            })

        output_item = {
            "id": q["id"],
            "question": q["question"],
            "question_type": q.get("question_type", ""),
            "difficulty": q.get("difficulty", ""),

            "retrieved": formatted_results,

            "gold_answer": q.get("gold_answer", ""),
            "gold_parent_doc_id": q.get("gold_parent_doc_id", ""),
            "gold_chunk_ids": q.get("gold_chunk_ids", [])
        }

        all_outputs.append(output_item)

    with open("retrieved_results.json", "w", encoding="utf-8") as f:
        json.dump(all_outputs, f, ensure_ascii=False, indent=2)

    print("All retrieval results saved to retrieval_results.json")





