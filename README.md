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

## `ingestion.py`

✓ PyPDFLoader          — loads PDF
✓ RecursiveCharacter   — advanced chunking (chunk_size + chunk_overlap)
✓ HuggingFaceEmbeddings — creates embedding model
✓ BM25Encoder          — learns vocabulary + word frequencies → saves bm25.json
✓ PineconeVectorStore  — ingests dense vectors into Pinecone

## `retriever.py`

✓ load_retriever()
  → same BAAI embedding model
  → dense + sparse Pinecone index variables
  → BM25Encoder loads bm25.json vocabulary
  → CrossEncoder reranker loaded

✓ retrieve()
  → query → BAAI → dense vector
  → query → BM25 → sparse vector
  → dense vector → searches dense index
  → sparse vector → searches sparse index
  → RRF combines both ranked lists
    (alpha * 1/(rank+60) for dense)
    (1-alpha * 1/(rank+60) for sparse)
    chunks in both lists get highest score
  → cross encoder reranks top k*2
  → returns top k chunks sorted descending

## `assistant.py`

✓ LLM — ChatGroq (llama-3.1-8b-instant)

✓ Prompt template — 3 parts:
  → system message (context + instructions)
  → MessagesPlaceholder (history)
  → human message (query)

✓ ChatMessageHistory — stores conversation

✓ chat() function:
  → retrieve() → get top 3 chunks
  → format_context() → join chunks into string
  → prompt_template.format_messages() → fill context + history + question
  → llm.invoke() → get answer
  → add_user_message() → save query to history
  → add_ai_message() → save answer to history
  → return answer

✓ __main__:
  → load_retriever() → load all components
  → while True loop → keep asking query
  → calls chat() → prints answer
  → breaks when user types "exit"


## Output

