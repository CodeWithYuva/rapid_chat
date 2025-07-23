from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def get_embedding(text):
    return model.encode(text).tolist()

def get_batch_embeddings(chunks):
    return [get_embedding(chunk["text"]) for chunk in chunks]
