"""
AuraGTM — Manual Client Ingestion Utility

Usage:
    python client_ingest.py <ClientName>
    python client_ingest.py <ClientName> --rebuild

Examples:
    python client_ingest.py BeamData
    python client_ingest.py Microsoft --rebuild
"""

import os
import sys
import argparse

from rag_pipeline import ingest

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run(client_name: str, rebuild: bool = False):
    client_folder = os.path.join(BASE_DIR, "knowledge_base","clients", client_name)
    db_folder     = os.path.join(BASE_DIR, "vector_db", "clients", f"{client_name}_db")
    hash_registry = os.path.join(BASE_DIR, "vector_db", "clients", f"{client_name}_ingested.json")

    if not os.path.exists(client_folder):
        print(f"❌  Client folder not found: {client_folder}")
        print(f"    Create it and add documents there before ingesting.")
        sys.exit(1)

    os.makedirs(db_folder, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Client : {client_name}")
    print(f"  Source : {client_folder}")
    print(f"  DB     : {db_folder}")
    print(f"  Rebuild: {rebuild}")
    print(f"{'='*60}\n")

    ingest(
        source_folders=[client_folder],
        db_directory=db_folder,
        hash_registry=hash_registry,
        rebuild=rebuild
    )

    # Report chunk count
    try:
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings
        _emb = OpenAIEmbeddings(model="text-embedding-3-small")
        _db  = Chroma(persist_directory=db_folder, embedding_function=_emb)
        count = _db._collection.count()
        print(f"\n✅  Done! [{client_name}] vector DB contains {count} chunks.")
    except Exception as e:
        print(f"\n⚠️   Could not verify chunk count: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AuraGTM Manual Client Ingestor")
    parser.add_argument("client_name", help="Client name (must match folder under 08_Clients_Data/)")
    parser.add_argument("--rebuild", action="store_true", help="Wipe and rebuild the vector DB from scratch")
    args = parser.parse_args()

    run(args.client_name, rebuild=args.rebuild)
