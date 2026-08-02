# 🧠 AI Memory OS

AI Memory OS is a fully offline, privacy-first **multimodal personal knowledge vault** and RAG (Retrieval-Augmented Generation) system. It allows you to index files, documents, spreadsheets, logs, and even images (using OCR) into a local vector database, and then search and chat with your accumulated knowledge using a local LLM. 

This project runs 100% locally on your machine, ensuring your data never leaves your computer.

---

## ✨ Features

- **🔒 Offline & Private**: Uses local embeddings (`all-MiniLM-L6-v2`) and local LLM execution via Ollama (defaulting to `llama3.1:latest`).
- **📥 Multimodal Ingestion Pipeline**: Extracts text from a wide range of file types, including:
  - **Documents**: `.pdf`, `.docx`, `.pptx`, `.txt`, `.md`, `.rtf`, `.log`
  - **Data/Spreadsheets**: `.xlsx`, `.xls`, `.csv`, `.json`, `.xml`
  - **Web**: `.html`, `.htm`
  - **Images (OCR)**: `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.tiff`, `.webp` (extracts text using Tesseract OCR and appends resolution/metadata).
- **🎨 Premium Streamlit Web GUI**: A modern, Instagram-inspired dark mode interface that supports:
  - **Memory Search**: Q&A using RAG with local LLM responses and semantic match scoring.
  - **Drag-and-Drop Ingester**: Upload any supported file types directly through the web app.
  - **Daily Journaling**: Write and immediately index daily notes and ideas.
  - **Memory Manager**: View stats (total chunks, indexed files), delete specific files from memory, or wipe the vector store completely.
- **💻 Interactive CLI Chat**: A fast terminal-based chat interface (`chat.py`) built with `rich` for quick command-line access to your memories.
- **📝 Daily Note Quick Capture**: A command-line utility (`daily_note.py`) to log quick thoughts directly into dated text files.

---

## 🛠️ Tech Stack

- **Framework**: [Streamlit](https://streamlit.io/)
- **Vector Database**: [ChromaDB](https://www.trychroma.com/)
- **Embedding Model**: [SentenceTransformers](https://sbert.net/) (`all-MiniLM-L6-v2`)
- **LLM Engine**: [Ollama](https://ollama.com/) (`llama3.1:latest`)
- **Document Processing**: `pypdf`, `python-docx`, `python-pptx`, `openpyxl`, `beautifulsoup4`
- **OCR Engine**: `pytesseract` & `Pillow`
- **Terminal UI**: [Rich](https://github.com/Textualize/rich)

---

## 🚀 Getting Started

### 📋 Prerequisites

1. **Python**: Version `3.10` or higher.
2. **Ollama**: Download and install [Ollama](https://ollama.com/). Once installed, pull the LLM:
   ```bash
   ollama pull llama3.1:latest
   ```
3. **Tesseract OCR** (Optional, for image OCR support):
   - **Ubuntu/Debian**:
     ```bash
     sudo apt update
     sudo apt install tesseract-ocr
     ```
   - **macOS**:
     ```bash
     brew install tesseract
     ```

### ⚙️ Installation

1. Clone the repository to your local machine:
   ```bash
   git clone <your-github-repo-url>
   cd local-RAG
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📖 Usage Instructions

### 1. Ingestion Pipeline
To import files into your memory store, place your documents or images in the `data/` directory and run:
```bash
python ingest.py
```
This script will parse your files, split them into chunks, generate embeddings, and store them in the ChromaDB collection.

### 2. Streamlit Web Interface
Start the premium web dashboard:
```bash
streamlit run app.py
```
This opens the browser interface where you can query your memories, upload new files directly, write in your daily journal, or view memory statistics.

### 3. Interactive CLI Chat
For a terminal-native conversational Q&A:
```bash
python chat.py
```

### 4. Daily Note Quick Capture
To log a quick thought via the command-line:
```bash
python daily_note.py
```
After writing, simply run `python ingest.py` to index the new journal entry.

---

## 📂 Project Structure

```text
local-RAG/
├── .streamlit/
│   └── config.toml        # Custom dark mode styling and typography
├── chroma_db/             # Persistent vector store database
├── data/                  # Directory for storing raw files & journal notes
├── app.py                 # Main Streamlit web application
├── chat.py                # Rich-based terminal chat application
├── ingest.py              # Main ingestion, OCR, and embedding script
├── daily_note.py          # Daily note quick capture CLI utility
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```
