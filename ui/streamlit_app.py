"""Streamlit UI for the PDF RAG agent. Run with:

    uv run streamlit run ui/streamlit_app.py

(Make sure the FastAPI backend is running first.)
"""
import requests
import streamlit as st

API = "http://localhost:8000"

st.set_page_config(page_title="PDF RAG Agent", page_icon="📄", layout="wide")
st.title("📄 Company AI-Initiatives — RAG Agent")

# --- Sidebar: index management ---------------------------------------------
with st.sidebar:
    st.header("Index")
    try:
        health = requests.get(f"{API}/health", timeout=5).json()
        st.caption(f"Device: **{health['device']}**")
        st.caption(f"Embed: `{health['embed_model']}`")
        st.caption(f"LLM: `{health['gen_model']}`")
    except Exception:
        st.error("Backend not reachable. Start FastAPI first.")

    if st.button("Ingest / Re-index PDFs", use_container_width=True):
        with st.spinner("Reading, embedding & indexing the PDFs..."):
            r = requests.post(f"{API}/ingest", timeout=600)
        st.json(r.json())

    st.divider()
    uploaded = st.file_uploader("Add more PDFs", type="pdf", accept_multiple_files=True)
    if uploaded and st.button("Upload", use_container_width=True):
        files = [("files", (f.name, f.getvalue(), "application/pdf")) for f in uploaded]
        r = requests.post(f"{API}/upload", files=files, timeout=120)
        st.success(f"Uploaded: {r.json().get('saved')} — now click Re-index.")

# --- Main: ask questions ----------------------------------------------------
st.subheader("Ask a question about the companies' AI initiatives")
question = st.text_input("Question", placeholder="e.g. How is NVIDIA using AI in its data centers?")
col1, col2 = st.columns([1, 3])
with col1:
    top_k = st.slider("Passages", 1, 10, 4)

if st.button("Ask", type="primary") and question:
    with st.spinner("Retrieving and generating..."):
        r = requests.post(
            f"{API}/query", json={"question": question, "top_k": top_k}, timeout=600
        )
    data = r.json()

    st.markdown("### Answer")
    st.write(data.get("answer"))

    st.markdown("### Sources")
    for i, s in enumerate(data.get("sources", []), 1):
        with st.expander(f"[{i}] {s['source']} — chunk {s['chunk']} (distance {s['distance']:.3f})"):
            st.write(s["text"])
