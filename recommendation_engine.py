import json, os
import faiss
import numpy as np
import cohere
import streamlit as st

cohere_api_key = os.getenv("COHERE_API_KEY")
if not cohere_api_key:
    raise ValueError("COHERE_API_KEY not found in environment variables.")


client = cohere.Client(cohere_api_key)

EMBED_MODEL = "embed-english-v3.0"

def load_catalog(path="data/shl_catalogue.json"):
    with open(path, "r") as f:
        return json.load(f)

def embed_texts(texts):
    res = client.embed(
        texts=texts,
        model=EMBED_MODEL,
        input_type="search_document"
    )
    return np.array(res.embeddings).astype("float32")

def build_faiss_index(catalog, index_path="faiss_index/index.faiss"):
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        with open("faiss_index/id_map.json", "r") as f:
            id_map = json.load(f)
        return index, {int(k): v for k, v in id_map.items()}

    os.makedirs("faiss_index", exist_ok=True)

    # Build description strings from relevant fields
    descriptions = [
        f"""Assessment Name: {item['Assessment Name']}
URL: {item['URL']}
Remote Support: {item['Remote Support']}
IRT Support: {item['IRT Support']}
Duration: {item['Duration']}
Test Type: {item['Test Type']}""" for item in catalog
    ]

    vectors = embed_texts(descriptions)

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    id_map = {i: catalog[i] for i in range(len(catalog))}
    faiss.write_index(index, index_path)
    with open("faiss_index/id_map.json", "w") as f:
        json.dump(id_map, f)

    return index, id_map

def retrieve_relevant_items(prompt, index, id_map, top_k=3):
    query_vec = embed_texts([prompt])[0].reshape(1, -1)
    D, I = index.search(query_vec, top_k)
    return [id_map[i] for i in I[0] if i in id_map]
