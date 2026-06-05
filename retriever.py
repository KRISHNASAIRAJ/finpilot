import os
from pinecone import Pinecone
from dotenv import load_dotenv
from ingestion import embedding
from sentence_transformers import CrossEncoder
from pinecone_text.sparse import BM25Encoder

load_dotenv()

def load_retriever():
    embeddings=embedding()
    reranker=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pc=Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    dense_index=pc.Index(os.environ["INDEX_NAME"])
    sparse_index=pc.Index(os.environ["SPARSE_INDEX_NAME"])
    bm25=BM25Encoder()
    bm25.load("bm25.json")
    print("Retriever loaded successfully.")
    return dense_index,sparse_index,bm25,embeddings,reranker

def retrieve(query,dense_index, sparse_index, bm25, embeddings, reranker, top_k=3, alpha=0.5):
    #embed the query for dense search
    dense_vector=embeddings.embed_query(query)
    #embed the query for sparse search
    sparse_vector=bm25.encode_queries([query])[0]
    #search in dense index
    dense_results=dense_index.query(vector=dense_vector, top_k=top_k*2, include_metadata=True)
    #search in sparse index
    sparse_results=sparse_index.query(sparse_vector=sparse_vector, top_k=top_k*2, include_metadata=True)
    
    #RRF Merging
    rrf_scores = {}
    all_matches = {}

    for rank, match in enumerate(dense_results["matches"]):
        rrf_scores[match["id"]] = rrf_scores.get(match["id"], 0) + alpha * (1 / (rank + 60))
        all_matches[match["id"]] = match

    for rank, match in enumerate(sparse_results["matches"]):
        rrf_scores[match["id"]] = rrf_scores.get(match["id"], 0) + (1 - alpha) * (1 / (rank + 60))
        all_matches[match["id"]] = match
    
    # step 6 — sort by RRF score get top k*2 where both dense and sparse scores are already combined in rrf_scores
    # so here we are storing id's of highest rrf_scores based on the combined score of dense and sparse search, we take top k*2 to have enough candidates for re-ranking in the next step
    sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k * 2]
    top_chunks = [
        all_matches[id]["metadata"]["text"]
        for id in sorted_ids
        if id in all_matches
    ]

    ''' step 7 — re-rank with cross-encoder since the order in hybrid search is not perfect, we can use a cross-encoder to re-rank the top candidates
    where cross-encoder gives score based on the relevance of the query and the chunk, we then sort the chunks based on this relevance score to get the final top k chunks that are most relevant to the query '''
    
    pairs = [[query, chunk] for chunk in top_chunks]
    scores = reranker.predict(pairs)

     # step 8 — return top k after reranking
    ranked = sorted(zip(scores, top_chunks), reverse=True)
    final_chunks = [chunk for _, chunk in ranked[:top_k]]
    return final_chunks

def format_context(chunks):
    return "\n\n".join(chunks)
