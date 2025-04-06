from fastapi import FastAPI
from recommendation_engine import load_catalog, build_faiss_index, retrieve_relevant_items

app = FastAPI()

catalog = load_catalog()
index, id_map = build_faiss_index(catalog)

@app.get("/query")
def query(q: str):
    results = retrieve_relevant_items(q, index, id_map)
    return results
