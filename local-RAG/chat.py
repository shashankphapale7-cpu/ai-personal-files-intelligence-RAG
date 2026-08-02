"""
AI Memory OS — Terminal Chat Interface
Interactive Q&A using ChromaDB + Ollama for RAG-based answers.
"""

import subprocess
import json
from pathlib import Path

import chromadb
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from sentence_transformers import SentenceTransformer

# ─── Config ───────────────────────────────────────────────────────────
CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "memories"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.1:latest"
TOP_K = 5

console = Console()


def get_collection():
    """Get or fail on the ChromaDB collection."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        return client.get_collection(name=COLLECTION_NAME)
    except Exception:
        console.print("[red]❌ No memories found. Run ingest.py first![/red]")
        raise SystemExit(1)


def search_memories(query: str, model: SentenceTransformer, collection, top_k: int = TOP_K):
    """Embed query and search ChromaDB for relevant chunks."""
    query_embedding = model.encode([query])[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return results


def build_prompt(query: str, context_chunks: list[str], sources: list[dict]) -> str:
    """Build RAG prompt with memory context."""
    context = "\n\n---\n\n".join(
        f"[From: {src.get('source', 'unknown')}]\n{chunk}"
        for chunk, src in zip(context_chunks, sources)
    )
    return f"""You are an AI assistant with access to the user's personal memory/knowledge base. 
Answer the question using ONLY the provided context. If the context doesn't contain enough information, say so honestly.
Be concise but thorough. Cite which source files the information came from.

### Context (from user's memory):
{context}

### Question:
{query}

### Answer:"""


def call_ollama(prompt: str) -> str:
    """Call Ollama via subprocess and return the response."""
    try:
        result = subprocess.run(
            ["ollama", "run", OLLAMA_MODEL],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            return f"Error calling Ollama: {result.stderr}"
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "⏱ Ollama response timed out (120s). Try a simpler question."
    except FileNotFoundError:
        return "❌ Ollama not found. Make sure it's installed and in PATH."
    except Exception as e:
        return f"Error: {e}"


def chat():
    """Main chat loop."""
    console.print(Panel(
        "[bold cyan]🧠 AI Memory OS — Chat[/bold cyan]\n"
        "[dim]Ask questions about your indexed memories.\n"
        "Type 'quit' or 'exit' to leave.[/dim]",
        border_style="bright_magenta",
    ))

    # Load model & collection
    console.print("[cyan]Loading embedding model...[/cyan]")
    model = SentenceTransformer(EMBEDDING_MODEL)
    collection = get_collection()
    console.print(f"[green]✓ Ready! {collection.count()} memories loaded.[/green]\n")

    while True:
        try:
            query = console.input("[bold magenta]You > [/bold magenta]").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            break

        # Search
        results = search_memories(query, model, collection)
        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []

        if not documents:
            console.print("[yellow]No relevant memories found.[/yellow]\n")
            continue

        # Show sources
        console.print("[dim]📎 Sources found:[/dim]")
        seen_sources = set()
        for meta, dist in zip(metadatas, distances):
            src = meta.get("source", "unknown")
            if src not in seen_sources:
                relevance = max(0, (1 - dist) * 100)
                console.print(f"  [dim]• {src} ({relevance:.0f}% relevant)[/dim]")
                seen_sources.add(src)

        # Build prompt & get answer
        prompt = build_prompt(query, documents, metadatas)
        console.print("[cyan]🤔 Thinking...[/cyan]")
        answer = call_ollama(prompt)

        console.print()
        console.print(Panel(
            Markdown(answer),
            title="[bold green]🤖 AI Memory[/bold green]",
            border_style="green",
        ))
        console.print()


if __name__ == "__main__":
    chat()
