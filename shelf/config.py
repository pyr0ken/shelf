from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
COVERS_DIR = DATA_DIR / "covers"
MODELS_CACHE_DIR = DATA_DIR / "models_cache"
DB_PATH = DATA_DIR / "shelf.db"

# Load local .env if available
env_file = BASE_DIR / ".env"
if env_file.exists():
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

BOOKS_DIR = Path(os.environ.get("SHELF_BOOKS_DIR", "/mnt/data/books"))

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

import socket

# Auto-configure proxy from environment if defined
def configure_proxy():
    os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
    socks_port = os.environ.get("SHELF_SOCKS_PORT")
    if socks_port and not os.environ.get("HTTP_PROXY") and not os.environ.get("ALL_PROXY"):
        try:
            with socket.create_connection(("127.0.0.1", int(socks_port)), timeout=0.1):
                os.environ["HTTP_PROXY"] = f"socks5h://127.0.0.1:{socks_port}"
                os.environ["HTTPS_PROXY"] = f"socks5h://127.0.0.1:{socks_port}"
                os.environ["ALL_PROXY"] = f"socks5h://127.0.0.1:{socks_port}"
        except (OSError, ValueError):
            pass

configure_proxy()

# LLM Gateway settings: Read from environment or fallback to standard local endpoints
def detect_llm_base_url() -> str:
    if "SHELF_LLM_BASE_URL" in os.environ:
        return os.environ["SHELF_LLM_BASE_URL"]
    for port in [8000, 11434]:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                return f"http://127.0.0.1:{port}/v1"
        except OSError:
            pass
    return "http://127.0.0.1:8000/v1"

LLM_BASE_URL = detect_llm_base_url()
LLM_API_KEY = os.environ.get("SHELF_LLM_API_KEY", "dummy-key")
LLM_MODEL = os.environ.get("SHELF_LLM_MODEL", "auto/best-fast")

DATA_DIR.mkdir(parents=True, exist_ok=True)
COVERS_DIR.mkdir(parents=True, exist_ok=True)
