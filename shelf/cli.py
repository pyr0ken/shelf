import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.markdown import Markdown

from shelf.config import BOOKS_DIR, DB_PATH
from shelf.db import (
    init_db,
    get_all_books,
    get_book,
    get_categories,
    update_reading_progress
)
from shelf.scanner import scan_single_book
from shelf.rag import index_book_rag, search_hybrid, generate_rag_answer

app = typer.Typer(
    help="Shelf: Intelligent Local-First Technical Library & RAG Engine",
    no_args_is_help=True
)
console = Console()

@app.command()
def scan(
    dir: Path = typer.Option(BOOKS_DIR, "--dir", "-d", help="Directory containing books to scan")
):
    """Scan books directory, extract covers, and index metadata into SQLite."""
    init_db()
    if not dir.exists():
        console.print(f"[red]Error:[/red] Books directory does not exist: {dir}")
        raise typer.Exit(1)

    all_files = [
        p for p in dir.rglob("*")
        if p.is_file() and p.suffix.lower() in [".pdf", ".epub"]
    ]

    console.print(f"[cyan]Found {len(all_files)} books in {dir}. Starting scan...[/cyan]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[green]Scanning & Extracting Covers...", total=len(all_files))
        scanned_count = 0
        for p in all_files:
            progress.update(task, description=f"[green]Processing:[/green] {p.name[:35]}...")
            res = scan_single_book(p, base_books_dir=dir)
            if res:
                scanned_count += 1
            progress.advance(task)

    console.print(f"[bold green]✓ Successfully scanned and indexed {scanned_count} books![/bold green]")

@app.command("list")
def list_books(
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Filter by category"),
    limit: int = typer.Option(50, "--limit", "-l", help="Number of books to display"),
    offset: int = typer.Option(0, "--offset", "-o", help="Offset for pagination")
):
    """List indexed books in the library."""
    init_db()
    books = get_all_books(category=category, limit=limit, offset=offset)
    if not books:
        console.print("[yellow]No books found. Run `shelf scan` first.[/yellow]")
        return

    table = Table(title=f"Shelf Books ({category or 'All Categories'})", header_style="bold magenta")
    table.add_column("ID", justify="right", style="cyan", no_wrap=True)
    table.add_column("Category", style="green")
    table.add_column("Title", style="bold white")
    table.add_column("Pages", justify="right", style="yellow")
    table.add_column("Fmt", justify="center", style="blue")
    table.add_column("RAG", justify="center", style="magenta")

    for b in books:
        rag_status = "✓" if b["is_indexed_rag"] else "-"
        table.add_row(
            str(b["id"]),
            b["category"],
            b["title"][:50],
            str(b["page_count"] or "?"),
            b["format"].upper(),
            rag_status
        )

    console.print(table)
    console.print(f"[dim]Showing {len(books)} books (offset {offset}).[/dim]")

@app.command()
def categories():
    """List all categories and book counts."""
    init_db()
    cats = get_categories()
    if not cats:
        console.print("[yellow]No categories found. Run `shelf scan` first.[/yellow]")
        return

    table = Table(title="Book Categories", header_style="bold cyan")
    table.add_column("Category", style="bold green")
    table.add_column("Count", justify="right", style="yellow")

    for c in cats:
        table.add_row(c["category"], str(c["count"]))

    console.print(table)

@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Filter by category"),
    limit: int = typer.Option(20, "--limit", "-l", help="Max search results")
):
    """Instant full-text search across book titles and categories."""
    init_db()
    books = get_all_books(category=category, query=query, limit=limit)
    if not books:
        console.print(f"[yellow]No books matching '{query}'.[/yellow]")
        return

    table = Table(title=f"Search Results for '{query}'", header_style="bold cyan")
    table.add_column("ID", justify="right", style="cyan")
    table.add_column("Category", style="green")
    table.add_column("Title", style="bold white")
    table.add_column("Format", style="blue")

    for b in books:
        table.add_row(str(b["id"]), b["category"], b["title"], b["format"].upper())

    console.print(table)

@app.command("index-rag")
def index_rag(
    book_id: Optional[int] = typer.Option(None, "--id", "-i", help="Index a single book ID"),
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Index all books in category"),
    all_books: bool = typer.Option(False, "--all", "-a", help="Index all books"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-indexing of already indexed books")
):
    """Index book text chunks and compute vector embeddings for RAG."""
    init_db()
    targets = []
    if book_id:
        b = get_book(book_id)
        if not b:
            console.print(f"[red]Book ID {book_id} not found.[/red]")
            raise typer.Exit(1)
        targets = [b]
    elif category:
        targets = get_all_books(category=category, limit=500)
    elif all_books:
        targets = get_all_books(limit=1000)
    else:
        console.print("[yellow]Please specify --id, --category, or --all to index for RAG.[/yellow]")
        raise typer.Exit(1)

    to_process = targets if force else [b for b in targets if not b["is_indexed_rag"]]
    skipped = len(targets) - len(to_process)

    if skipped > 0:
        console.print(f"[dim]Skipping {skipped} already indexed book(s) (use --force to re-index).[/dim]")

    if not to_process:
        console.print("[bold green]✓ All target books are already indexed for RAG![/bold green]")
        return

    console.print(f"[cyan]Targeting {len(to_process)} book(s) for RAG indexing...[/cyan]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[magenta]Embedding chunks...", total=len(to_process))
        total_chunks = 0
        for b in to_process:
            progress.update(task, description=f"[magenta]Embedding:[/magenta] {b['title'][:30]}...")
            cnt = index_book_rag(b["id"])
            total_chunks += cnt
            progress.advance(task)

    console.print(f"[bold green]✓ RAG indexing complete! Created and embedded {total_chunks} new chunks.[/bold green]")

@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask your books library"),
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Restrict search to category"),
    book_id: Optional[int] = typer.Option(None, "--book-id", "-b", help="Restrict search to single book"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of retrieved chunks"),
    no_llm: bool = typer.Option(False, "--no-llm", "-n", help="Skip LLM synthesis and only show raw book excerpts & citations")
):
    """Ask your technical books a question with verified citations and page references."""
    init_db()
    with console.status("[bold cyan]Searching library & synthesizing answer...[/bold cyan]"):
        result = generate_rag_answer(
            question,
            category=category,
            book_id=book_id,
            top_k=top_k,
            use_llm=not no_llm
        )

    console.print(Panel(
        Markdown(result["answer"]),
        title=f"[bold green]Answer to: {question}[/bold green]",
        border_style="green"
    ))

    if result.get("sources"):
        console.print("\n[bold cyan]📚 Grounded Citations & Sources:[/bold cyan]")
        for i, s in enumerate(result["sources"], 1):
            ch = f" | [yellow]{s['chapter_title']}[/yellow]" if s.get("chapter_title") else ""
            console.print(f" [dim]{i}.[/dim] [bold white]{s['book_title']}[/bold white] - Page [bold yellow]{s['page_number']}[/bold yellow]{ch} [dim](RRF: {s.get('rrf_score', 0):.4f})[/dim]")
            snippet = s["content"][:200].replace("\n", " ").strip()
            console.print(f"    [italic dim]\"{snippet}...\"[/italic dim]\n")

@app.command("open")
@app.command("open-book")
def open_book(
    book_id: int = typer.Argument(..., help="Book ID to open"),
    page: Optional[int] = typer.Option(None, "--page", "-p", help="Page number to jump to")
):
    """Open book in desktop reader (Sioyek, Zathura, or default) and track progress."""
    init_db()
    b = get_book(book_id)
    if not b:
        console.print(f"[red]Book ID {book_id} not found.[/red]")
        raise typer.Exit(1)

    filepath = b["filepath"]
    target_page = page or b["current_page"] or 1

    console.print(f"[green]Opening '{b['title']}' at page {target_page}...[/green]")
    update_reading_progress(book_id, target_page)

    # Detect installed reader
    readers = [
        ("sioyek", ["sioyek", f"--page={target_page}", filepath]),
        ("zathura", ["zathura", f"-P", str(target_page), filepath]),
        ("okular", ["okular", f"-p", str(target_page), filepath]),
        ("xdg-open", ["xdg-open", filepath])
    ]

    launched = False
    for name, cmd in readers:
        if shutil.which(name):
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            console.print(f"[cyan]Launched with {name}.[/cyan]")
            launched = True
            break

    if not launched:
        subprocess.Popen(["xdg-open", filepath], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        console.print("[cyan]Launched with default system viewer.[/cyan]")

@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Bind host"),
    port: int = typer.Option(7788, "--port", "-p", help="Bind port")
):
    """Launch the Web Dashboard & RAG interface."""
    import uvicorn
    console.print(f"[bold green]Starting Shelf Web UI at http://{host}:{port}[/bold green]")
    uvicorn.run("shelf.server:app", host=host, port=port, reload=False)

if __name__ == "__main__":
    app()
