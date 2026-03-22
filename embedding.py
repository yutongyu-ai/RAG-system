import os
import json
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np

client = OpenAI()  # Initialize OpenAI API client for embeddings

bge_base_model = SentenceTransformer("BAAI/bge-base-en")  # Load BGE-base model
bge_large_model = SentenceTransformer("BAAI/bge-large-en")  # Load BGE-large model

def get_embeddings(texts, model_name):
    """
    Generate the embedding vector for chunks using the specified model.
    """
    if model_name == "openai":
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [d.embedding for d in response.data]

    elif model_name == "bge-base":
        texts = ["Represent this sentence for retrieval: " + t for t in texts]
        return bge_base_model.encode(texts, normalize_embeddings=True)

    elif model_name == "bge-large":
        texts = ["Represent this sentence for retrieval: " + t for t in texts]
        return bge_large_model.encode(texts, normalize_embeddings=True)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def embed_chunks(chunks, model_name="openai", batch_size=32):
    """
    Convert chunks into embeddings and store them with metadata
    """
    vector_data = []

    for i in tqdm(range(0, len(chunks), batch_size)):
        batch = chunks[i:i+batch_size]

        inputs = [c["text"] for c in batch]

        embeddings = get_embeddings(inputs, model_name)

        for j, emb in enumerate(embeddings):
            vector_data.append({
                "id": batch[j]["chunk_id"],
                "embedding": np.array(emb).tolist(),
                "text": batch[j]["text"],
                "metadata": {
                    "parent_doc_id": batch[j]["parent_doc_id"],
                    "source": batch[j]["source"]
                }
            })

    return vector_data


if __name__ == "__main__":
    with open("chunked_data.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    vector_data = embed_chunks(chunks, model_name="openai")

    folder_path = "vector_store"
    file_path = os.path.join(folder_path, "vector_store_openai.json")
    os.makedirs(folder_path, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(vector_data, f, ensure_ascii=False)

    print("Saved to vector_store")