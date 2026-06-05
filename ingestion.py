import os
import warnings
from dotenv import load_dotenv
from pinecone_text.sparse import BM25Encoder
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()
warnings.filterwarnings("ignore")

def load_pdf(file_path):
    loader=PyPDFLoader(file_path)
    document=loader.load()
    print("Document loaded successfully.")
    return document

def chunking(document,chunk_size,chunk_overlap):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["\n\n","\n",".","!","?"," ", ""]
    )
    chunks=text_splitter.create_documents([doc.page_content for doc in document])
    print("Document chunked successfully.")
    return chunks

def embedding():
    embeddings=HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    print("Embeddings generated successfully.")
    return embeddings

def fit_bm25_encoder(chunks):
    texts=[chunk.page_content for chunk in chunks]
    bm25=BM25Encoder()
    bm25.fit(texts)
    bm25.dump("bm25.json")
    print("BM25 encoder fitted and saved successfully.")

def ingestion(chunks,embeddings):
    PineconeVectorStore.from_documents(chunks,embeddings,index_name=os.environ["INDEX_NAME"])
    print("Ingestion completed successfully.")

if __name__ == "__main__":
    file_path="data/report.pdf"
    document=load_pdf(file_path)
    chunks=chunking(document,1000,100)
    fit_bm25_encoder(chunks)
    embeddings=embedding()
    ingestion(chunks,embeddings)