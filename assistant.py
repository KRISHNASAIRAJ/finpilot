import os
import warnings
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from retriever import load_retriever, retrieve, format_context

load_dotenv()
warnings.filterwarnings("ignore")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.environ.get("GROQ_API_KEY")
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are FinPilot — a personal portfolio assistant.
    Answer honestly and without bias based only on the following context:
    {context}"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

chat_history = ChatMessageHistory()

def chat(query, dense_index, sparse_index, bm25, embeddings, reranker):
    chunks = retrieve(query, dense_index, sparse_index, bm25, embeddings, reranker)
    context = format_context(chunks)
    messages = prompt_template.format_messages(
        context=context,
        history=chat_history.messages,
        question=query
    )
    response = llm.invoke(messages)
    answer = response.content
    chat_history.add_user_message(query)
    chat_history.add_ai_message(answer)
    return answer

if __name__ == "__main__":
    print("Loading FinPilot...")
    dense_index, sparse_index, bm25, embeddings, reranker = load_retriever()
    print("\n🚀 FinPilot ready! Type 'exit' to quit.\n")
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            print("Thanks for using!")
            break
        answer = chat(query, dense_index, sparse_index, bm25, embeddings, reranker)
        print(f"\nFinPilot: {answer}\n")