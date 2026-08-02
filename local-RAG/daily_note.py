"""
AI Memory OS — Daily Note Quick Capture
"""

import datetime
from pathlib import Path
from rich.console import Console

DATA_DIR = Path(__file__).parent / "data"
console = Console()

def add_daily_note():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().strftime("%Y-%m-%d")
    note_file = DATA_DIR / f"{today}.txt"

    console.print(f"[bold cyan]📝 Daily Note Quick Capture ({today})[/bold cyan]")
    console.print("[dim]Enter your note below (Ctrl+D or empty line to finish):[/dim]\n")

    lines = []
    while True:
        try:
            line = input("> ")
            if not line.strip() and len(lines) > 0:
                break
            lines.append(line)
        except EOFError:
            break

    content = "\n".join(lines).strip()
    if not content:
        console.print("[yellow]Empty note. Nothing saved.[/yellow]")
        return

    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    formatted_entry = f"\n--- [{timestamp}] ---\n{content}\n"

    with open(note_file, "a", encoding="utf-8") as f:
        f.write(formatted_entry)

    console.print(f"[bold green]✓ Note saved to {note_file.name}![/bold green]")
    console.print("[dim]Run `python ingest.py` to index this new note into memory.[/dim]")

if __name__ == "__main__":
    add_daily_note()
