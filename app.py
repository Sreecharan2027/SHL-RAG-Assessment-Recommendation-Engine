import streamlit as st
import json
from recommendation_engine import load_catalog, build_faiss_index, retrieve_relevant_items
from llm_utils import build_prompt, query_llm


st.set_page_config(page_title="SHL RAG Engine", page_icon="🧠")
st.title("🔍 SHL RAG Assessment Recommendation Engine")

# Load catalog and FAISS index
catalog = load_catalog()
index, id_map = build_faiss_index(catalog)

st.markdown("Describe the role or hiring need. We'll recommend matching SHL assessments 🔍")

# Text input + Submit trigger
query = st.text_input("📝 Your Input", placeholder="e.g., Hiring a junior developer with coding and problem-solving skills")

st.markdown("##### 💡 Example queries you can try:")
examples = [
    "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes.",
    "Looking to hire mid-level professionals who are proficient in Python, SQL and Java Script. Need an assessment package that can test all skills with max duration of 60 minutes.",
    "I am hiring for an analyst and wants applications to screen using Cognitive and personality tests, what options are available within 45 mins."
]
for example in examples:
    st.markdown(f"- {example}")

# Button or Enter triggers recommendation
if st.button("🔎 Recommend via RAG") or (query and st.session_state.get("last_query") != query):
    if query.strip():
        with st.spinner("Retrieving and reasoning..."):
            st.session_state["last_query"] = query
            retrieved = retrieve_relevant_items(query, index, id_map, top_k=3)
            rag_prompt = build_prompt(query, retrieved)
            result = query_llm(rag_prompt)

        st.success("🧠 LLM Recommendation")
        st.markdown(result)

        st.markdown("---")
        st.subheader("📚 Retrieved Context")
        for item in retrieved:
            st.markdown(f"### 📝 {item['Assessment Name']}")
            st.markdown(f"- **Duration**: {item['Duration']}")
            st.markdown(f"- **Test Type**: `{item['Test Type']}`")
            st.markdown(f"- **Remote Support**: {item['Remote Support']}")
            st.markdown(f"- **IRT Support**: {item['IRT Support']}")
            st.markdown(f"- **URL**: [View Assessment]({item['URL']})")
            st.markdown("---")

        