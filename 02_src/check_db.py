"""
AuraGTM — quick database diagnostic.
Run from the project root (with venv active):  python check_db.py
"""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "vector_db", "global_db")

print("DB path :", DB)
print("Exists  :", os.path.exists(DB))

emb = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(persist_directory=DB, embedding_function=emb)

count = db._collection.count()
print("Total chunks in global_db:", count)

print("\n--- 1) Plain similarity search: 'retail consumer Saudi Arabia' ---")
try:
    res = db.similarity_search("retail consumer Saudi Arabia", k=3)
    print("results:", len(res))
    for d in res:
        print("   topic =", d.metadata.get("topic"), "| file =", d.metadata.get("filename"))
except Exception as e:
    print("PLAIN SEARCH ERROR:", repr(e))

print("\n--- 2) Filtered search on topic '03_Market_Knowledge' ---")
try:
    res2 = db.similarity_search("retail", k=3, filter={"topic": {"$eq": "03_Market_Knowledge"}})
    print("results:", len(res2))
    for d in res2:
        print("   file =", d.metadata.get("filename"))
except Exception as e:
    print("FILTERED SEARCH ERROR:", repr(e))

print("\n--- 3) MMR + filter (EXACTLY what the engine uses) ---")
try:
    res3 = db.max_marginal_relevance_search(
        query="retail", k=3, fetch_k=10,
        filter={"topic": {"$eq": "03_Market_Knowledge"}}
    )
    print("results:", len(res3))
    for d in res3:
        print("   file =", d.metadata.get("filename"))
except Exception as e:
    print("MMR ERROR:", repr(e))

print("\n--- 4) Distinct topics actually stored in the DB ---")
try:
    md = db._collection.get(include=["metadatas"])["metadatas"]
    topics = sorted(set((m or {}).get("topic") for m in md))
    print(topics)
except Exception as e:
    print("METADATA ERROR:", repr(e))