from fastapi import FastAPI, Request
from recommendation_engine import retrieve_relevant_items

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "API is running!"}

@app.get("/query")
def query_api(q: str):
    result = retrieve_relevant_items(q)
    return {"result": result}
