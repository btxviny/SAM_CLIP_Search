import chromadb
import torch
import clip
import numpy as np
import os 
import random


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
print(f"Loaded CLIP onto device: {device}")

#Create Vector Store Collection
def create_vectorstore(collection_name: str, embeddings_path: str):
    """
    Create a new vector store collection from a directory of embeddings.

    Parameters
    ----------
    collection_name : str
        The name of the collection to create.
    embeddings_path : str
        The path to the directory containing the embeddings.

    Notes
    -----
    If the collection already exists, it will be overwritten.
    The embeddings are assumed to be in numpy format with the same name as the object id.
    The object id is assumed to be the filename without the extension.
    Random metadata is generated for the embeddings.
    """
    chroma_client = chromadb.PersistentClient(path="./vector_store/")
    # check if collection already exists and overwrite it
    for collection in chroma_client.list_collections():
        if collection.name == collection_name:
            chroma_client.delete_collection(collection.name)

    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    for embedding_file in os.listdir(embeddings_path):
        embedding = np.load(os.path.join(embeddings_path, embedding_file))
        collection.add(
            documents=[embedding_file],
            embeddings=[embedding],
            metadatas=[
                {
                    "filename": embedding_file,
                    "object_id": embedding_file[:-4],
                    "xc": random.uniform(0,15),
                    "yc": random.uniform(0,15),
                    "zc": random.uniform(0,15),
                }
            ],
            ids=[embedding_file[:-4]]
        )

def embed_text(text: str, model):
    """
    Encode a text string into a vector using CLIP.

    Parameters
    ----------
    text : str
        The text string to encode.
    model : torch.nn.Module
        The CLIP model to use for encoding.

    Returns
    -------
    text_features : numpy.ndarray
        The encoded vector.
    """
    text = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text).cpu().numpy().flatten()
    return text_features

def query_vectorstore(query: str, collection):
    # chroma_client = chromadb.PersistentClient(path="./vector_store/")
    # collection = chroma_client.get_collection(collection_name)
    """
    Query a vector store collection with a text string to find the closest matching items.

    Parameters
    ----------
    query : str
        The text string to query the vector store with.
    collection : chromadb.Collection
        The collection to query against.

    Returns
    -------
    dict
        Metadata of the closest matching items in the collection.
    """
    text_features = embed_text(query, model)
    results = collection.query(
        query_embeddings=text_features,
        n_results=3
    )
    return {
        "object_id": results["metadatas"][0][0]["object_id"],
        "filename": results["metadatas"][0][0]["filename"],
        "xc": results["metadatas"][0][0]["xc"],
        "yc": results["metadatas"][0][0]["yc"],
        "zc": results["metadatas"][0][0]["zc"],
    }