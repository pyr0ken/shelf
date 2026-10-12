import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from shelf.config import DB_PATH

def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

def init_db(db_path: Path = DB_PATH):
    conn = get_connection(db_path)
    cur = conn.cursor()

    cur.executescript("""
    CREATE TABLE IF NOT EXISTS books (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        filename TEXT NOT NULL,
        filepath TEXT UNIQUE NOT NULL,
        category TEXT NOT NULL,
        subcategory TEXT,
        format TEXT NOT NULL,
        file_size INTEGER,
        page_count INTEGER,
        cover_path TEXT,
        toc_json TEXT,
        current_page INTEGER DEFAULT 1,
        last_read_at DATETIME,
        is_indexed_rag INTEGER DEFAULT 0,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        book_id INTEGER NOT NULL REFERENCES books(id) ON DELETE CASCADE,
        page_number INTEGER NOT NULL,
        chapter_title TEXT,
        chunk_index INTEGER NOT NULL,
        content TEXT NOT NULL,
        token_count INTEGER,
        embedding BLOB
    );

    CREATE INDEX IF NOT EXISTS idx_chunks_book_id ON chunks(book_id);
    CREATE INDEX IF NOT EXISTS idx_chunks_page ON chunks(book_id, page_number);

    -- Full-Text Search on books
    CREATE VIRTUAL TABLE IF NOT EXISTS books_fts USING fts5(
        title,
        category,
        subcategory,
        filename,
        content='books',
        content_rowid='id'
    );

    -- Full-Text Search on chunks
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
        content,
        chapter_title,
        content='chunks',
        content_rowid='id'
    );
    """)

    # Triggers for books_fts
    cur.executescript("""
    CREATE TRIGGER IF NOT EXISTS books_ai AFTER INSERT ON books BEGIN
        INSERT INTO books_fts(rowid, title, category, subcategory, filename)
        VALUES (new.id, new.title, new.category, new.subcategory, new.filename);
    END;

    CREATE TRIGGER IF NOT EXISTS books_ad AFTER DELETE ON books BEGIN
        INSERT INTO books_fts(books_fts, rowid, title, category, subcategory, filename)
        VALUES('delete', old.id, old.title, old.category, old.subcategory, old.filename);
    END;

    CREATE TRIGGER IF NOT EXISTS books_au AFTER UPDATE ON books BEGIN
        INSERT INTO books_fts(books_fts, rowid, title, category, subcategory, filename)
        VALUES('delete', old.id, old.title, old.category, old.subcategory, old.filename);
        INSERT INTO books_fts(rowid, title, category, subcategory, filename)
        VALUES (new.id, new.title, new.category, new.subcategory, new.filename);
    END;
    """)

    # Triggers for chunks_fts
    cur.executescript("""
    CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
        INSERT INTO chunks_fts(rowid, content, chapter_title)
        VALUES (new.id, new.content, new.chapter_title);
    END;

    CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts, rowid, content, chapter_title)
        VALUES('delete', old.id, old.content, old.chapter_title);
    END;

    CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts, rowid, content, chapter_title)
        VALUES('delete', old.id, old.content, old.chapter_title);
        INSERT INTO chunks_fts(rowid, content, chapter_title)
        VALUES (new.id, new.content, new.chapter_title);
    END;
    """)

    conn.commit()
    conn.close()

def upsert_book(
    filepath: str,
    title: str,
    filename: str,
    category: str,
    subcategory: Optional[str],
    file_format: str,
    file_size: int,
    page_count: int,
    cover_path: Optional[str] = None,
    toc: Optional[List[Dict[str, Any]]] = None,
    db_path: Path = DB_PATH
) -> int:
    conn = get_connection(db_path)
    cur = conn.cursor()
    toc_json = json.dumps(toc, ensure_ascii=False) if toc else None

    cur.execute("""
    INSERT INTO books (
        filepath, title, filename, category, subcategory,
        format, file_size, page_count, cover_path, toc_json, updated_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    ON CONFLICT(filepath) DO UPDATE SET
        title=excluded.title,
        filename=excluded.filename,
        category=excluded.category,
        subcategory=excluded.subcategory,
        file_size=excluded.file_size,
        page_count=excluded.page_count,
        cover_path=COALESCE(excluded.cover_path, books.cover_path),
        toc_json=COALESCE(excluded.toc_json, books.toc_json),
        updated_at=CURRENT_TIMESTAMP
    RETURNING id;
    """, (filepath, title, filename, category, subcategory, file_format, file_size, page_count, cover_path, toc_json))

    book_id = cur.fetchone()[0]
    conn.commit()
    conn.close()
    return book_id

def get_book(book_id: int, db_path: Path = DB_PATH) -> Optional[Dict[str, Any]]:
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute("SELECT * FROM books WHERE id = ?", (book_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None

def get_all_books(
    category: Optional[str] = None,
    query: Optional[str] = None,
    limit: int = 500,
    offset: int = 0,
    db_path: Path = DB_PATH
) -> List[Dict[str, Any]]:
    conn = get_connection(db_path)
    cur = conn.cursor()

    if query:
        # FTS on books
        sql = """
        SELECT b.* FROM books_fts f
        JOIN books b ON f.rowid = b.id
        WHERE books_fts MATCH ?
        """
        params: List[Any] = [query]
        if category:
            sql += " AND b.category = ?"
            params.append(category)
        sql += " ORDER BY rank LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        cur.execute(sql, params)
    else:
        sql = "SELECT * FROM books WHERE 1=1"
        params: List[Any] = []
        if category:
            sql += " AND category = ?"
            params.append(category)
        sql += " ORDER BY category ASC, title ASC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        cur.execute(sql, params)

    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_categories(db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute("""
    SELECT category, COUNT(*) as count 
    FROM books 
    GROUP BY category 
    ORDER BY count DESC, category ASC
    """)
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def update_reading_progress(book_id: int, page_number: int, db_path: Path = DB_PATH):
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute("""
    UPDATE books 
    SET current_page = ?, last_read_at = CURRENT_TIMESTAMP
    WHERE id = ?
    """, (page_number, book_id))
    conn.commit()
    conn.close()

def save_chunks(
    book_id: int,
    chunks_data: List[Dict[str, Any]],
    db_path: Path = DB_PATH
):
    conn = get_connection(db_path)
    cur = conn.cursor()

    # Clear existing chunks for this book
    cur.execute("DELETE FROM chunks WHERE book_id = ?", (book_id,))

    insert_rows = []
    for c in chunks_data:
        embedding_blob = None
        if "embedding" in c and c["embedding"] is not None:
            emb = np.array(c["embedding"], dtype=np.float32)
            embedding_blob = emb.tobytes()

        insert_rows.append((
            book_id,
            c["page_number"],
            c.get("chapter_title"),
            c["chunk_index"],
            c["content"],
            c.get("token_count", len(c["content"].split())),
            embedding_blob
        ))

    cur.executemany("""
    INSERT INTO chunks (
        book_id, page_number, chapter_title, chunk_index,
        content, token_count, embedding
    ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, insert_rows)

    cur.execute("UPDATE books SET is_indexed_rag = 1 WHERE id = ?", (book_id,))
    conn.commit()
    conn.close()

def get_chunks_with_embeddings(
    category: Optional[str] = None,
    book_id: Optional[int] = None,
    db_path: Path = DB_PATH
) -> List[Dict[str, Any]]:
    conn = get_connection(db_path)
    cur = conn.cursor()

    sql = """
    SELECT c.id, c.book_id, c.page_number, c.chapter_title, c.chunk_index,
           c.content, c.embedding, b.title as book_title, b.category, b.filepath
    FROM chunks c
    JOIN books b ON c.book_id = b.id
    WHERE c.embedding IS NOT NULL
    """
    params = []
    if book_id is not None:
        sql += " AND c.book_id = ?"
        params.append(book_id)
    elif category:
        sql += " AND b.category = ?"
        params.append(category)

    cur.execute(sql, params)
    rows = cur.fetchall()
    conn.close()

    results = []
    for r in rows:
        d = dict(r)
        if d["embedding"]:
            d["embedding"] = np.frombuffer(d["embedding"], dtype=np.float32)
        results.append(d)
    return results

def search_chunks_fts(
    query: str,
    category: Optional[str] = None,
    book_id: Optional[int] = None,
    limit: int = 20,
    db_path: Path = DB_PATH
) -> List[Dict[str, Any]]:
    conn = get_connection(db_path)
    cur = conn.cursor()

    sql = """
    SELECT c.id, c.book_id, c.page_number, c.chapter_title, c.chunk_index,
           c.content, b.title as book_title, b.category, b.filepath,
           f.rank as fts_rank
    FROM chunks_fts f
    JOIN chunks c ON f.rowid = c.id
    JOIN books b ON c.book_id = b.id
    WHERE chunks_fts MATCH ?
    """
    params: List[Any] = [query]
    if book_id is not None:
        sql += " AND c.book_id = ?"
        params.append(book_id)
    elif category:
        sql += " AND b.category = ?"
        params.append(category)

    sql += " ORDER BY rank LIMIT ?"
    params.append(limit)

    cur.execute(sql, params)
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]
