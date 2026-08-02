"""
AI Memory OS — Ingestion Pipeline
Reads files from data/, extracts text, chunks, embeds, and stores in ChromaDB.
Supports: txt, pdf, docx, xlsx, xls, csv, md, json, xml, html, rtf, log, pptx, images (png, jpg, jpeg, gif, bmp, tiff, webp)
"""

import os
import sys
import csv
import json
import hashlib
import datetime
from pathlib import Path
from html.parser import HTMLParser

import chromadb
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from sentence_transformers import SentenceTransformer

# ─── Config ───────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "memories"
CHUNK_SIZE = 500
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Supported file extensions
TEXT_EXTENSIONS = {".txt", ".md", ".json", ".xml", ".rtf", ".log"}
DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".pptx"}
SPREADSHEET_EXTENSIONS = {".xlsx", ".xls", ".csv"}
WEB_EXTENSIONS = {".html", ".htm"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"}

ALL_SUPPORTED = TEXT_EXTENSIONS | DOCUMENT_EXTENSIONS | SPREADSHEET_EXTENSIONS | WEB_EXTENSIONS | IMAGE_EXTENSIONS

console = Console()


# ─── HTML Text Stripper ──────────────────────────────────────────────
class HTMLTextExtractor(HTMLParser):
    """Strip HTML tags and extract plain text."""
    def __init__(self):
        super().__init__()
        self._result = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            self._result.append(data)

    def get_text(self):
        return " ".join(self._result)


# ─── File Readers ─────────────────────────────────────────────────────
def read_text_file(filepath: Path) -> str:
    """Read plain text files."""
    try:
        encodings = ["utf-8", "latin-1", "cp1252"]
        for enc in encodings:
            try:
                return filepath.read_text(encoding=enc)
            except UnicodeDecodeError:
                continue
        # Fallback: read as bytes and decode with errors='replace'
        return filepath.read_bytes().decode("utf-8", errors="replace")
    except Exception as e:
        console.print(f"  [red]Error reading {filepath.name}: {e}[/red]")
        return ""


def read_pdf(filepath: Path) -> str:
    """Extract text from PDF using pypdf."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(filepath))
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        return "\n".join(text_parts)
    except Exception as e:
        console.print(f"  [red]Error reading PDF {filepath.name}: {e}[/red]")
        return ""


def read_docx(filepath: Path) -> str:
    """Extract text from .docx using python-docx."""
    try:
        from docx import Document
        doc = Document(str(filepath))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        # Also read tables
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells)
                if row_text.strip():
                    paragraphs.append(row_text)
        return "\n".join(paragraphs)
    except Exception as e:
        console.print(f"  [red]Error reading DOCX {filepath.name}: {e}[/red]")
        return ""


def read_pptx(filepath: Path) -> str:
    """Extract text from .pptx using python-pptx."""
    try:
        from pptx import Presentation
        prs = Presentation(str(filepath))
        text_parts = []
        for slide_num, slide in enumerate(prs.slides, 1):
            slide_texts = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_texts.append(shape.text.strip())
            if slide_texts:
                text_parts.append(f"[Slide {slide_num}]\n" + "\n".join(slide_texts))
        return "\n\n".join(text_parts)
    except Exception as e:
        console.print(f"  [red]Error reading PPTX {filepath.name}: {e}[/red]")
        return ""


def read_xlsx(filepath: Path) -> str:
    """Extract text from .xlsx/.xls using openpyxl."""
    try:
        from openpyxl import load_workbook
        wb = load_workbook(str(filepath), read_only=True, data_only=True)
        text_parts = []
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            rows = list(ws.iter_rows(values_only=True))
            if not rows:
                continue
            text_parts.append(f"[Sheet: {sheet_name}]")
            # First row as headers
            headers = [str(h) if h is not None else "" for h in rows[0]]
            for row in rows[1:]:
                row_items = []
                for i, val in enumerate(row):
                    if val is not None:
                        header = headers[i] if i < len(headers) else f"Col{i}"
                        row_items.append(f"{header}: {val}")
                if row_items:
                    text_parts.append(", ".join(row_items))
        wb.close()
        return "\n".join(text_parts)
    except Exception as e:
        console.print(f"  [red]Error reading XLSX {filepath.name}: {e}[/red]")
        return ""


def read_csv_file(filepath: Path) -> str:
    """Extract text from CSV."""
    try:
        text_parts = []
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            rows = list(reader)
            if not rows:
                return ""
            headers = rows[0]
            for row in rows[1:]:
                row_items = []
                for i, val in enumerate(row):
                    if val.strip():
                        header = headers[i] if i < len(headers) else f"Col{i}"
                        row_items.append(f"{header}: {val}")
                if row_items:
                    text_parts.append(", ".join(row_items))
        return "\n".join(text_parts)
    except Exception as e:
        console.print(f"  [red]Error reading CSV {filepath.name}: {e}[/red]")
        return ""


def read_html(filepath: Path) -> str:
    """Extract text from HTML files."""
    try:
        raw = filepath.read_text(encoding="utf-8", errors="replace")
        extractor = HTMLTextExtractor()
        extractor.feed(raw)
        return extractor.get_text()
    except Exception as e:
        console.print(f"  [red]Error reading HTML {filepath.name}: {e}[/red]")
        return ""


def read_image(filepath: Path) -> str:
    """Extract text from images using OCR (pytesseract) and return description."""
    text_parts = []
    text_parts.append(f"[Image: {filepath.name}]")

    # Try OCR
    try:
        from PIL import Image
        import pytesseract
        img = Image.open(filepath)
        ocr_text = pytesseract.image_to_string(img).strip()
        if ocr_text:
            text_parts.append(f"OCR Text: {ocr_text}")
    except ImportError:
        text_parts.append("(OCR not available — install pytesseract and Tesseract engine)")
    except Exception as e:
        text_parts.append(f"(OCR failed: {e})")

    # Basic image metadata
    try:
        from PIL import Image
        img = Image.open(filepath)
        w, h = img.size
        text_parts.append(f"Dimensions: {w}x{h}, Format: {img.format}, Mode: {img.mode}")
    except Exception:
        pass

    return "\n".join(text_parts)


def extract_text(filepath: Path) -> str:
    """Route file to the appropriate reader based on extension."""
    ext = filepath.suffix.lower()

    if ext in TEXT_EXTENSIONS:
        return read_text_file(filepath)
    elif ext == ".pdf":
        return read_pdf(filepath)
    elif ext == ".docx":
        return read_docx(filepath)
    elif ext == ".pptx":
        return read_pptx(filepath)
    elif ext in (".xlsx", ".xls"):
        return read_xlsx(filepath)
    elif ext == ".csv":
        return read_csv_file(filepath)
    elif ext in WEB_EXTENSIONS:
        return read_html(filepath)
    elif ext in IMAGE_EXTENSIONS:
        return read_image(filepath)
    else:
        return ""


# ─── Chunking ─────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """Split text into chunks of approximately `chunk_size` characters."""
    if not text.strip():
        return []
    chunks = []
    words = text.split()
    current_chunk = []
    current_len = 0
    for word in words:
        word_len = len(word) + 1  # +1 for space
        if current_len + word_len > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_len = word_len
        else:
            current_chunk.append(word)
            current_len += word_len
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


# ─── Unique ID Generation ────────────────────────────────────────────
def make_chunk_id(filename: str, chunk_index: int) -> str:
    """Generate a deterministic, unique ID for each chunk."""
    raw = f"{filename}::chunk_{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


# ─── Main Ingestion ──────────────────────────────────────────────────
def ingest(data_dir: Path = DATA_DIR, show_progress: bool = True):
    """
    Main ingestion pipeline.
    Returns dict with stats: {files_processed, chunks_created, errors}
    """
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
        console.print(f"[yellow]Created data directory: {data_dir}[/yellow]")
        console.print("[yellow]Add some files to data/ and run again.[/yellow]")
        return {"files_processed": 0, "chunks_created": 0, "errors": []}

    # Collect supported files
    files = [f for f in data_dir.iterdir() if f.is_file() and f.suffix.lower() in ALL_SUPPORTED]
    if not files:
        console.print("[yellow]No supported files found in data/[/yellow]")
        return {"files_processed": 0, "chunks_created": 0, "errors": []}

    console.print(f"\n[bold cyan]🧠 AI Memory OS — Ingestion Pipeline[/bold cyan]")
    console.print(f"[dim]Found {len(files)} file(s) to process[/dim]\n")

    # Load embedding model
    console.print("[cyan]Loading embedding model...[/cyan]")
    model = SentenceTransformer(EMBEDDING_MODEL)
    console.print(f"[green]✓ Model loaded: {EMBEDDING_MODEL}[/green]\n")

    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    stats = {"files_processed": 0, "chunks_created": 0, "errors": []}
    all_ids = []
    all_docs = []
    all_embeddings = []
    all_metadatas = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
        disable=not show_progress,
    ) as progress:
        task = progress.add_task("Processing files...", total=len(files))

        for filepath in files:
            progress.update(task, description=f"Processing {filepath.name}...")

            text = extract_text(filepath)
            if not text.strip():
                stats["errors"].append(f"{filepath.name}: No text extracted")
                progress.advance(task)
                continue

            chunks = chunk_text(text)
            if not chunks:
                stats["errors"].append(f"{filepath.name}: No chunks created")
                progress.advance(task)
                continue

            # Generate embeddings for all chunks at once
            embeddings = model.encode(chunks).tolist()

            timestamp = datetime.datetime.now().isoformat()
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = make_chunk_id(filepath.name, i)
                all_ids.append(chunk_id)
                all_docs.append(chunk)
                all_embeddings.append(embedding)
                all_metadatas.append({
                    "source": filepath.name,
                    "file_type": filepath.suffix.lower(),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "indexed_at": timestamp,
                })

            stats["files_processed"] += 1
            stats["chunks_created"] += len(chunks)
            progress.advance(task)

    # Upsert all at once (handles re-ingestion gracefully)
    if all_ids:
        # ChromaDB has a batch limit, so upsert in batches
        batch_size = 500
        for i in range(0, len(all_ids), batch_size):
            end = min(i + batch_size, len(all_ids))
            collection.upsert(
                ids=all_ids[i:end],
                documents=all_docs[i:end],
                embeddings=all_embeddings[i:end],
                metadatas=all_metadatas[i:end],
            )

    console.print(f"\n[bold green]✅ Ingestion Complete![/bold green]")
    console.print(f"  📁 Files processed: [cyan]{stats['files_processed']}[/cyan]")
    console.print(f"  🧩 Chunks created:  [cyan]{stats['chunks_created']}[/cyan]")
    console.print(f"  💾 Collection size: [cyan]{collection.count()}[/cyan] total memories")

    if stats["errors"]:
        console.print(f"\n  [yellow]⚠ Warnings:[/yellow]")
        for err in stats["errors"]:
            console.print(f"    [dim]{err}[/dim]")

    return stats


if __name__ == "__main__":
    ingest()
