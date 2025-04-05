# api.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from recommendation_engine import load_catalog, build_faiss_index, retrieve_relevant_items

app = FastAPI()

# Load catalog and build FAISS index
catalog = load_catalog()
index, id_map = build_faiss_index(catalog)

# Request body model
class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

@app.post("/recommend")
def recommend_assessments(request: QueryRequest):
    results = retrieve_relevant_items(request.query, index, id_map, top_k=request.top_k)
    return {"results": results}
