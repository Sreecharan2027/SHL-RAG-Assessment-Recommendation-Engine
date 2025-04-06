from fastapi import FastAPI
from recommendation_engine import retrieve_relevant_items

app = FastAPI()

@app.get("/")
def home():
    return {"message": "SHL RAG API is up and running!"}

@app.get("/query")
def get_recommendation(q: str):
    result = retrieve_relevant_items(q)
    return {"result": result}
