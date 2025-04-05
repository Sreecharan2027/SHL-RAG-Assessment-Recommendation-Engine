import os
from dotenv import load_dotenv
import cohere

load_dotenv()
cohere_api_key = os.getenv("COHERE_API_KEY")
co = cohere.Client(cohere_api_key)

def build_prompt(query, items):
    context = "\n".join([
        f"""Assessment Name: {i['Assessment Name']}
URL: {i['URL']}
Remote Support: {i['Remote Support']}
IRT Support: {i['IRT Support']}
Duration: {i['Duration']}
Test Type: {i['Test Type']}
""" for i in items])

    return f"""You are an expert in SHL assessments.

Based on the following context, recommend assessments that match the user query.

Context:
{context}

User Query:
{query}

Instructions:
Respond with a list of recommended assessments with reasons in markdown format.
"""

def query_llm(prompt):
    response = co.chat(
        message=prompt,
        model="command-r",
        temperature=0.4
    )
    return response.text
