# finpilot


## V1 Features
* Multi-PDF ingestion
* Hybrid search retrieval
* Re-ranking
* Conversation memory
* Terminal chat loop

## Future Versions

* Chainlit UI/Android App
* Mutual fund analysis
* Credit card comparison
* Budget tracking
* Deployment

#### Libraries to be downloaded
* `pip install uv`

* `uv init`

    Note: Here we are using uv as package manager since it is fast as compared to pip since it is designed using Rust.

* `uv add langchain-community langchain-huggingface langchain-pinecone langchain-groq langchain-text-splitters langchain-core langchain-experimental pinecone-text sentence-transformers pypdf python-dotenv`

## BM25 (Hybrid Search)

* So basically it is a scoring algorithm for the keywords. how many times do the query words appear in this chunk, and how important are those words?

* It converts chunks into sparse vectors. Notice rare words score higher. 

* So during ingestion phase it Learns vocabulary + frequencies of the words which is called `fitting`. Phase 1 — fit (ingestion)

* bm25.fit(chunks)  ← reads ALL your chunks

* So when we pass a query then it will go for `scoring` and returns the most relavant one's. Phase 2 — score (query time)


            Ingestion                         Retrieval
Dense   chunk → BAAI → store            query → BAAI → search
Sparse  chunk → BM25 → store            query → BM25 → search