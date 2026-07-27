"""
AuraGTM - Document Ingestion Pipeline
"""

import os
import shutil
import hashlib
import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

try:
    from langchain_community.document_loaders import Docx2txtLoader
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False
    print("⚠️  Docx2txtLoader not available. Run: pip install docx2txt")

CHUNK_SIZE     = 1000
CHUNK_OVERLAP  = 200
EMBEDDING_MODEL = "text-embedding-3-small"


def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_hash_registry(registry_path: str) -> dict:
    if os.path.exists(registry_path):
        with open(registry_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_hash_registry(registry: dict, registry_path: str):
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)


def derive_topic_from_path(file_path: str, base_folder: str) -> str:
    rel = os.path.relpath(file_path, base_folder)
    parts = Path(rel).parts
    if len(parts) > 1:
        return parts[0]
    return Path(parts[0]).stem


def load_file(file_path: str, base_folder: str) -> list:
    ext = Path(file_path).suffix.lower()
    topic = derive_topic_from_path(file_path, base_folder)
    filename = Path(file_path).name

    try:
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        elif ext in (".docx", ".doc") and DOCX_SUPPORT:
            loader = Docx2txtLoader(file_path)
        else:
            print(f"  ⏭  Skipping (unsupported type): {filename}")
            return []

        pages = loader.load()

        for page in pages:
            page.metadata["topic"]    = topic
            page.metadata["filename"] = filename
            page.metadata["source"]   = file_path

        has_text = any(p.page_content.strip() for p in pages)
        if not has_text:
            print(f"  ⚠️  No extractable text in: {filename} (might be a scanned image)")

        print(f"  ✅ Loaded {len(pages)} pages from [{topic}] {filename}")
        return pages

    except Exception as e:
        print(f"  ❌ Error loading {filename}: {e}")
        return []


def load_folder(folder_path: str, skip_registry: dict) -> tuple:
    print(f"\n📂 Scanning: {folder_path}")
    documents = []
    new_registry = dict(skip_registry)
    supported_ext = {".pdf", ".txt", ".docx", ".doc"}

    for root, _, files in os.walk(folder_path):
        for file in sorted(files):
            if Path(file).suffix.lower() not in supported_ext:
                continue

            file_path = os.path.join(root, file)
            md5 = file_md5(file_path)

            if file_path in skip_registry and skip_registry[file_path] == md5:
                print(f"  ⏩ Already processed: {file}")
                continue

            pages = load_file(file_path, folder_path)
            if pages:
                documents.extend(pages)
                new_registry[file_path] = md5

    return documents, new_registry


def split_text_into_chunks(documents: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    print(f"\n✂️  Created {len(chunks)} text chunks.")
    return chunks


def get_or_create_vector_db(db_directory: str, rebuild: bool = False) -> Chroma:
    if rebuild and os.path.exists(db_directory):
        shutil.rmtree(db_directory)
        print("🗑️  Old database deleted.")

    embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vector_db = Chroma(
        persist_directory=db_directory,
        embedding_function=embedding_model
    )
    return vector_db


def add_chunks_to_db(vector_db: Chroma, chunks: list):
    if not chunks:
        print("ℹ️  No new chunks to add.")
        return
    vector_db.add_documents(chunks)
    print(f"💾 Added {len(chunks)} chunks to the database.")


def ingest(source_folders: list, db_directory: str, hash_registry: str, rebuild: bool = False):
    print("=" * 60)
    print("  AuraGTM — Ingestion Pipeline")
    print("=" * 60)

    registry = {} if rebuild else load_hash_registry(hash_registry)
    all_new_docs = []

    for folder in source_folders:
        if not os.path.exists(folder):
            print(f"⚠️  Folder not found: {folder}")
            continue
        new_docs, registry = load_folder(folder, registry)
        all_new_docs.extend(new_docs)

    if not all_new_docs:
        print("\n✅ No new files found. Database is already up to date.")
        return

    chunks = split_text_into_chunks(all_new_docs)
    vector_db = get_or_create_vector_db(db_directory, rebuild=rebuild)
    add_chunks_to_db(vector_db, chunks)
    save_hash_registry(registry, hash_registry)

    topics = set(c.metadata.get("topic", "Unknown") for c in chunks)
    print(f"\n📌 Topics in database: {', '.join(sorted(topics))}")
    print(f"✅ Ingestion complete!\n")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    GLOBAL_KB = os.path.join(BASE_DIR, "knowledge_base", "global")

    # FIX: point ingestion at the PARENT global folder (not each subfolder
    # individually). This way every document's `topic` is set to its subfolder
    # name — e.g. "01_GTM_Frameworks" — which is exactly what the engine filters
    # on in _retrieve_from_topic(). Listing each subfolder separately made the
    # topic default to the file name instead, so global retrieval matched nothing.
    SOURCE_FOLDERS = [GLOBAL_KB]

    DATABASE_FOLDER = os.path.join(BASE_DIR, "vector_db", "global_db")
    HASH_REGISTRY = os.path.join(BASE_DIR, "global_ingested_files.json")

    ingest(
        SOURCE_FOLDERS,
        DATABASE_FOLDER,
        HASH_REGISTRY,
        rebuild=True
    )