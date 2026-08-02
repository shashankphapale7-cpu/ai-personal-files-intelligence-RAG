"""
AI Memory OS — Streamlit GUI
Includes Memory Search, Multimodal Ingestion & Delete Memory controls, Journal, and Memory Stats.
"""

import os
import shutil
import datetime
from pathlib import Path

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import subprocess

# ─── Config & Paths ───────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
CHROMA_DIR = ROOT_DIR / "chroma_db"
COLLECTION_NAME = "memories"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.1:latest"

DATA_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="AI Memory OS",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Load Model & Vector DB Cached ────────────────────────────────────
@st.cache_resource
def get_embed_model():
    return SentenceTransformer(EMBEDDING_MODEL)

def get_chroma_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

embed_model = get_embed_model()
collection = get_chroma_collection()

# ─── Helper Functions ─────────────────────────────────────────────────
def run_ollama_query(prompt: str) -> str:
    try:
        res = subprocess.run(
            ["ollama", "run", OLLAMA_MODEL],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=120
        )
        if res.returncode == 0:
            return res.stdout.strip()
        return f"Error: {res.stderr}"
    except Exception as e:
        return f"Execution error: {e}"

def delete_memory_by_source(filename: str):
    """Delete all chunks originating from a specific file."""
    results = collection.get(where={"source": filename})
    if results and results["ids"]:
        collection.delete(ids=results["ids"])
    
    file_path = DATA_DIR / filename
    if file_path.exists():
        os.remove(file_path)

def clear_all_memories():
    """Wipe out entire database and data folder."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    
    get_chroma_collection()
    
    for item in DATA_DIR.iterdir():
        if item.is_file():
            item.unlink()

# ─── Sidebar Navigation ────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 AI Memory OS")
    st.caption("Offline Multimodal Knowledge Vault")
    st.divider()

    page = st.radio(
        "Navigation",
        ["🔍 Memory Search", "📥 Ingest Files", "📝 Daily Journal", "📊 Memory Stats & Management"],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown(f"**Indexed Chunks:** `{collection.count()}`")
    st.markdown(f"**Active LLM:** `{OLLAMA_MODEL}`")

# ─── Page 1: Memory Search ──────────────────────────────────────────────
if page == "🔍 Memory Search":
    st.title("Search Your Memories")
    st.caption("Query across indexed documents, images, and notes.")

    query = st.text_input("Ask a question about your stored knowledge:", placeholder="e.g. What are my goals for Q3?")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        top_k = st.slider("Top Sources", 1, 10, 5)

    if query:
        with st.spinner("Searching memories & generating answer..."):
            query_emb = embed_model.encode([query])[0].tolist()
            search_results = collection.query(
                query_embeddings=[query_emb],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )

            docs = search_results["documents"][0] if search_results["documents"] else []
            metas = search_results["metadatas"][0] if search_results["metadatas"] else []
            dists = search_results["distances"][0] if search_results["distances"] else []

            if not docs:
                st.warning("No relevant memories found in database.")
            else:
                context_str = "\n\n---\n\n".join([f"[From {m.get('source', 'unknown')}]: {d}" for d, m in zip(docs, metas)])
                prompt = f"""You are an AI Memory Assistant. Answer using ONLY the provided memory context.
If uncertain, state so clearly.

Context:
{context_str}

Question: {query}
Answer:"""

                answer = run_ollama_query(prompt)

                # Display Answer in native bordered container
                with st.container(border=True):
                    st.subheader("🤖 Answer")
                    st.write(answer)

                st.subheader("📎 Source Memories")
                for doc, meta, dist in zip(docs, metas, dists):
                    relevance = max(0, (1 - dist) * 100)
                    src = meta.get("source", "Unknown File")
                    
                    with st.container(border=True):
                        c1, c2 = st.columns([3, 1])
                        with c1:
                            st.markdown(f"**📄 {src}**")
                        with c2:
                            st.markdown(f":green[{relevance:.0f}% Match]" if relevance > 60 else f":orange[{relevance:.0f}% Match]")
                        st.caption(doc)

# ─── Page 2: Ingest Files ──────────────────────────────────────────────
elif page == "📥 Ingest Files":
    st.title("Ingest & Index Files")
    st.caption("Upload multimodal documents (PDF, DOCX, XLSX, PPTX, Images, Text, etc.)")

    uploaded_files = st.file_uploader(
        "Upload files to add to AI Memory:",
        accept_multiple_files=True,
        type=["txt", "pdf", "docx", "pptx", "xlsx", "xls", "csv", "md", "json", "xml", "html", "png", "jpg", "jpeg", "bmp"]
    )

    if uploaded_files:
        if st.button("🚀 Start Ingestion", type="primary"):
            with st.spinner("Processing & embedding files..."):
                saved_count = 0
                for f in uploaded_files:
                    target_path = DATA_DIR / f.name
                    with open(target_path, "wb") as out_file:
                        out_file.write(f.read())
                    saved_count += 1
                
                from ingest import ingest
                stats = ingest(data_dir=DATA_DIR, show_progress=False)
                
                st.success(f"Successfully processed {stats['files_processed']} files and generated {stats['chunks_created']} memory chunks!")

# ─── Page 3: Daily Journal ─────────────────────────────────────────────
elif page == "📝 Daily Journal":
    st.title("Daily Journal")
    st.caption("Quickly capture thoughts and memory snippets for today.")

    today_str = datetime.date.today().strftime("%Y-%m-%d")
    journal_text = st.text_area(f"Journal Entry for {today_str}:", height=200, placeholder="Write your notes here...")

    if st.button("💾 Save & Index Note", type="primary"):
        if journal_text.strip():
            note_file = DATA_DIR / f"{today_str}.txt"
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            entry = f"\n--- [{timestamp}] ---\n{journal_text.strip()}\n"
            
            with open(note_file, "a", encoding="utf-8") as f:
                f.write(entry)
            
            from ingest import ingest
            ingest(data_dir=DATA_DIR, show_progress=False)
            st.success(f"Journal entry saved to `{note_file.name}` and indexed into memory!")
        else:
            st.warning("Journal entry cannot be empty.")

# ─── Page 4: Memory Stats & Management ──────────────────────────────────
elif page == "📊 Memory Stats & Management":
    st.title("Memory Stats & Management")
    st.caption("Inspect stored files, delete memories, or reset the memory store.")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Memories (Chunks)", collection.count())
    with col2:
        existing_files = list(DATA_DIR.iterdir()) if DATA_DIR.exists() else []
        st.metric("Total Indexed Files in Storage", len([f for f in existing_files if f.is_file()]))

    st.divider()
    st.subheader("🗑️ Delete Specific Memory File")
    
    if existing_files:
        file_options = [f.name for f in existing_files if f.is_file()]
        selected_file = st.selectbox("Select file to delete from AI Memory:", file_options)
        
        if st.button("🗑️ Delete Selected Memory"):
            delete_memory_by_source(selected_file)
            st.success(f"Deleted `{selected_file}` and all associated chunks from memory.")
            st.rerun()
    else:
        st.info("No files currently stored in memory.")

    st.divider()
    st.subheader("⚠️ Danger Zone")
    st.warning("Clearing all memories will permanently erase all indexed chunks and data files.")
    
    if st.button("🔥 Wipe Entire AI Memory Storage", type="primary"):
        clear_all_memories()
        st.success("All memories wiped clean!")
        st.rerun()
