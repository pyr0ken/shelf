import re
import io
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pymupdf as fitz
from PIL import Image

# Silence EPUB / font CSS warnings
fitz.TOOLS.mupdf_display_errors(False)

from shelf.config import BOOKS_DIR, COVERS_DIR
from shelf.db import upsert_book, save_chunks

def clean_book_title(filename: str, meta_title: Optional[str] = None) -> str:
    if meta_title and len(meta_title.strip()) > 4 and not meta_title.lower().startswith("untitled"):
        # Check if meta title is reasonably clean
        clean_meta = meta_title.strip()
        if not any(x in clean_meta.lower() for x in ["adobe indesign", "microsoft word", "tex output"]):
            return clean_meta

    stem = Path(filename).stem
    # Remove common junk suffixes
    stem = re.sub(r'[\(\[\{].*?(?:z-lib|libgen|ebooksworld|pdfdrive|springer|oreilly).*?[\)\]\}]', '', stem, flags=re.IGNORECASE)
    stem = re.sub(r'\.(?:pdf|epub|djvu)$', '', stem, flags=re.IGNORECASE)
    stem = stem.replace('_', ' ').replace('-', ' ')
    stem = re.sub(r'\s+', ' ', stem).strip()
    return stem or Path(filename).stem

def extract_cover(doc: fitz.Document, book_id: int) -> Optional[str]:
    try:
        if len(doc) == 0:
            return None
        page = doc[0]
        # Render at 1.5x scale for sharp WebP thumbnail
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        
        # Max dimensions: 600x900
        img.thumbnail((600, 900), Image.Resampling.LANCZOS)
        
        cover_filename = f"{book_id}.webp"
        cover_path = COVERS_DIR / cover_filename
        img.save(cover_path, "WEBP", quality=82)
        return cover_filename
    except Exception as e:
        return None

def extract_toc(doc: fitz.Document) -> List[Dict[str, Any]]:
    toc_raw = doc.get_toc()  # [[lvl, title, page], ...]
    toc = []
    for item in toc_raw:
        if len(item) >= 3:
            lvl, title, page = item[0], str(item[1]).strip(), int(item[2])
            if title and page > 0:
                toc.append({"level": lvl, "title": title, "page": page})
    return toc

def find_chapter_for_page(toc: List[Dict[str, Any]], page_num: int) -> Optional[str]:
    current_chapter = None
    for item in toc:
        if item["page"] <= page_num:
            current_chapter = item["title"]
        else:
            break
    return current_chapter

def scan_single_book(filepath: Path, base_books_dir: Path = BOOKS_DIR) -> Optional[Dict[str, Any]]:
    try:
        rel_path = filepath.relative_to(base_books_dir)
        parts = rel_path.parts
        
        category = parts[0] if len(parts) > 1 else "General"
        subcategory = parts[1] if len(parts) > 2 else None
        file_format = filepath.suffix.lower().lstrip(".")
        file_size = filepath.stat().st_size
        filename = filepath.name

        doc = fitz.open(str(filepath))
        page_count = len(doc)
        meta_title = doc.metadata.get("title") if doc.metadata else None
        title = clean_book_title(filename, meta_title)
        toc = extract_toc(doc)
        doc.close()

        # Temporary upsert to obtain book_id
        book_id = upsert_book(
            filepath=str(filepath),
            title=title,
            filename=filename,
            category=category,
            subcategory=subcategory,
            file_format=file_format,
            file_size=file_size,
            page_count=page_count,
            cover_path=None,
            toc=toc
        )

        # Extract and save cover now that we have book_id
        doc = fitz.open(str(filepath))
        cover_filename = extract_cover(doc, book_id)
        doc.close()

        if cover_filename:
            upsert_book(
                filepath=str(filepath),
                title=title,
                filename=filename,
                category=category,
                subcategory=subcategory,
                file_format=file_format,
                file_size=file_size,
                page_count=page_count,
                cover_path=cover_filename,
                toc=toc
            )

        return {
            "id": book_id,
            "title": title,
            "category": category,
            "subcategory": subcategory,
            "page_count": page_count,
            "cover": cover_filename
        }
    except Exception as e:
        return None

def chunk_book_text(
    filepath: Path,
    toc: Optional[List[Dict[str, Any]]] = None,
    chunk_size_words: int = 400,
    chunk_overlap_words: int = 80
) -> List[Dict[str, Any]]:
    chunks = []
    doc = fitz.open(str(filepath))
    toc = toc or extract_toc(doc)
    chunk_idx = 0

    for page_idx in range(len(doc)):
        page_num = page_idx + 1
        page = doc[page_idx]
        raw_text = page.get_text("text")
        text = str(raw_text).strip() if isinstance(raw_text, str) else ""
        if not text or len(text) < 50:
            continue

        chapter_title = find_chapter_for_page(toc, page_num)
        words = text.split()

        # If page is short enough, keep as single chunk
        if len(words) <= chunk_size_words:
            chunks.append({
                "page_number": page_num,
                "chapter_title": chapter_title,
                "chunk_index": chunk_idx,
                "content": text,
                "token_count": len(words)
            })
            chunk_idx += 1
        else:
            # Slide window across long page
            start = 0
            while start < len(words):
                end = min(start + chunk_size_words, len(words))
                chunk_words = words[start:end]
                chunk_text = " ".join(chunk_words)
                
                chunks.append({
                    "page_number": page_num,
                    "chapter_title": chapter_title,
                    "chunk_index": chunk_idx,
                    "content": chunk_text,
                    "token_count": len(chunk_words)
                })
                chunk_idx += 1
                if end == len(words):
                    break
                start += (chunk_size_words - chunk_overlap_words)

    doc.close()
    return chunks
