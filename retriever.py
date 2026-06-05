import os
from dotenv import load_dotenv
from ingestion import embedding
from sentence_transformers import CrossEncoder
from pinecone import Pinecone

load_dotenv()

def load_retriever():
    embeddings=embedding()
    reranker=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pc=Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    dense_index=pc.Index(os.environ["INDEX_NAME"])
    sparse_index=pc.Index(os.environ["SPARSE_INDEX_NAME"])
    bm25_index=BM25Encoder.load("bm25.json")
    print("Retriever loaded successfully.")
    return dense_index,sparse_index,bm25_index,embeddings,reranker

def retrive(query,dense_index, sparse_index, bm25_index, embeddings, reranker, top_k=3, alpha=0.5):
    pass