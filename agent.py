import os
import warnings
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent
from retriever import load_retriever, retrieve, format_context

load_dotenv()
warnings.filterwarnings("ignore")

# ── LLM ──
llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # larger model — better tool calling
    groq_api_key=os.environ.get("GROQ_API_KEY")
)

# ── Load retriever ──
print("Loading FinPilot Agent...")
dense_index, sparse_index, bm25, embeddings, reranker = load_retriever()

# ── Tool 1 — Portfolio Retriever ──
@tool
def portfolio_search(query: str) -> str:
    """Search the portfolio documents for questions about stocks,
    dividend yield, PE ratio, returns, NAV, occupancy rates."""
    chunks = retrieve(query, dense_index, sparse_index, bm25, embeddings, reranker)
    return format_context(chunks)

# ── Tool 2 — Calculator ──
@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression.
    Input should be a valid math expression like '25 * 34835 / 100'"""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

# ── Tool 3 — Web Search ──
search = DuckDuckGoSearchRun()

@tool
def web_search(query: str) -> str:
    """Search the web for latest stock news and live information."""
    try:
        return search.run(query)
    except Exception as e:
        return f"Search failed: {str(e)}"

# ── Create Agent ──
tools = [portfolio_search, calculator, web_search]
agent = create_react_agent(llm, tools)

if __name__ == "__main__":
    print("\n🚀 FinPilot Agent ready! Type 'exit' to quit.\n")
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            print("Goodbye!")
            break
        try:
            response = agent.invoke({
                "messages": [{"role": "user", "content": query}]
            })
            print(f"\nFinPilot: {response['messages'][-1].content}\n")
        except Exception as e:
            print(f"\nFinPilot: Error — {str(e)}\n")  # ← show actual error