import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from shelf.config import COVERS_DIR, BASE_DIR
from shelf.db import (
    init_db,
    get_all_books,
    get_book,
    get_categories,
    update_reading_progress,
    add_bookmark,
    delete_bookmark,
    get_bookmarks
)
from shelf.rag import generate_rag_answer, search_hybrid

app = FastAPI(title="Shelf API", version="0.1.0")

STATIC_DIR = BASE_DIR / "shelf" / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

class RagRequest(BaseModel):
    question: str
    category: Optional[str] = None
    book_id: Optional[int] = None
    top_k: int = 5
    use_llm: bool = True

class OpenRequest(BaseModel):
    page: Optional[int] = None
    reader: Optional[str] = None

class BookmarkCreateRequest(BaseModel):
    book_id: int
    page_number: int
    title: Optional[str] = None
    note: Optional[str] = None

@app.on_event("startup")
def on_startup():
    init_db()

@app.get("/api/books")
def list_books_api(
    category: Optional[str] = None,
    query: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    books = get_all_books(category=category, query=query, limit=limit, offset=offset)
    return {"books": books, "count": len(books)}

@app.get("/api/books/{book_id}")
def get_book_api(book_id: int):
    book = get_book(book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    return book

@app.get("/api/categories")
def get_categories_api():
    return {"categories": get_categories()}

@app.get("/api/covers/{filename}")
def get_cover_api(filename: str):
    cover_file = COVERS_DIR / filename
    if not cover_file.exists():
        raise HTTPException(status_code=404, detail="Cover not found")
    return FileResponse(str(cover_file), media_type="image/webp")

@app.get("/api/books/{book_id}/pdf")
def get_book_pdf_api(book_id: int):
    book = get_book(book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    filepath = Path(book["filepath"])
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="PDF file not found on disk")
    return FileResponse(
        str(filepath),
        media_type="application/pdf",
        headers={"Content-Disposition": "inline"}
    )

@app.post("/api/books/{book_id}/open")
def open_book_api(book_id: int, req: Optional[OpenRequest] = None):
    book = get_book(book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")

    filepath = book["filepath"]
    target_page = (req.page if req and req.page else None) or book["current_page"] or 1
    update_reading_progress(book_id, target_page)

    # If Chrome requested explicitly
    if req and req.reader == "chrome":
        chrome_bins = ["google-chrome-stable", "google-chrome", "chromium"]
        for b in chrome_bins:
            if shutil.which(b):
                subprocess.Popen([b, f"file://{filepath}#page={target_page}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return {"status": "ok", "reader": b, "page": target_page}

    readers = [
        ("google-chrome-stable", ["google-chrome-stable", f"file://{filepath}#page={target_page}"]),
        ("google-chrome", ["google-chrome", f"file://{filepath}#page={target_page}"]),
        ("sioyek", ["sioyek", f"--page={target_page}", filepath]),
        ("zathura", ["zathura", "-P", str(target_page), filepath]),
        ("okular", ["okular", "-p", str(target_page), filepath]),
        ("xdg-open", ["xdg-open", filepath])
    ]

    for name, cmd in readers:
        if shutil.which(name):
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return {"status": "ok", "reader": name, "page": target_page}

    subprocess.Popen(["xdg-open", filepath], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return {"status": "ok", "reader": "xdg-open", "page": target_page}

@app.get("/api/bookmarks")
def list_bookmarks_api(book_id: Optional[int] = None):
    return {"bookmarks": get_bookmarks(book_id=book_id)}

@app.post("/api/bookmarks")
def create_bookmark_api(req: BookmarkCreateRequest):
    bm_id = add_bookmark(
        book_id=req.book_id,
        page_number=req.page_number,
        title=req.title,
        note=req.note
    )
    return {"status": "ok", "bookmark_id": bm_id}

@app.delete("/api/bookmarks/{bookmark_id}")
def delete_bookmark_api(bookmark_id: int):
    success = delete_bookmark(bookmark_id)
    if not success:
        raise HTTPException(status_code=404, detail="Bookmark not found")
    return {"status": "ok"}

@app.post("/api/rag/ask")
def ask_rag_api(req: RagRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    result = generate_rag_answer(
        question=req.question,
        category=req.category,
        book_id=req.book_id,
        top_k=req.top_k,
        use_llm=req.use_llm
    )
    return result

@app.get("/")
def index():
    html_file = STATIC_DIR / "index.html"
    if html_file.exists():
        return FileResponse(str(html_file))
    return JSONResponse({"status": "Shelf backend active", "version": "0.1.0"})
