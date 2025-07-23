from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

qdrant = QdrantClient(host="localhost", port=6333)
collection_name = "rag_documents"

def setup_collection(vector_size):
    try:
        qdrant.get_collection(collection_name)
    except:
        qdrant.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )

def insert_documents(chunks, embeddings):
    setup_collection(len(embeddings[0]))
    points = [
        PointStruct(
            id=i,
            vector=embeddings[i],
            payload=chunks[i]["metadata"] | {"text": chunks[i]["text"]}
        )
        for i in range(len(chunks))
    ]
    qdrant.upsert(collection_name=collection_name, points=points)
    return len(points)

def search(query_embedding, top_k=3):
    results = qdrant.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k
    )
    return [
        {
            "text": r.payload["text"],
            "source": r.payload["source"],
            "page": r.payload.get("page"),
            "chunk_id": r.payload["chunk_id"]
        }
        for r in results
    ]
