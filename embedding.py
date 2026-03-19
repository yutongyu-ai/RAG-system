import json
from openai import OpenAI
from tqdm import tqdm

client = OpenAI()


def embed_chunks(chunks, batch_size=32):
    vector_data = []

    for i in tqdm(range(0, len(chunks), batch_size)):
        batch = chunks[i:i+batch_size]

        inputs = [c["text"] for c in batch]

        response = client.embeddings.create(
            model = "text-embedding-3-small",
            input=inputs
        )

        for j, emb in enumerate(response.data):
            vector_data.append({
                "id": batch[j]["chunk_id"],
                "embedding": emb.embedding,
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

    print(len(chunks))

    vector_data = embed_chunks(chunks)

    with open("vector_store.json", "w", encoding="utf-8") as f:
        json.dump(vector_data, f, ensure_ascii=False)

    print("Saved vector_store.json")