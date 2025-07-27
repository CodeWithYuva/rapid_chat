from sentence_transformers import SentenceTransformer
import os

local_model_path = "local_models/bge-small-en-v1.5"

# Check if offline model exists
if not os.path.isdir(local_model_path):
    load_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    load_model.save(local_model_path)


# Load locally
model = SentenceTransformer(local_model_path)

def get_embedding(text):
    return model.encode(text).tolist()

def get_batch_embeddings(chunks):
    return [get_embedding(chunk["text"]) for chunk in chunks]


def get_embedding(text):
    return model.encode(text).tolist()

def get_batch_embeddings(chunks):
    return [get_embedding(chunk["text"]) for chunk in chunks]
