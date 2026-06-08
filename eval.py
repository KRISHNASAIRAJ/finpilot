import os
import warnings
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from retriever import load_retriever, retrieve, format_context
from assistant import chat, chat_history

load_dotenv()
warnings.filterwarnings("ignore")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.environ.get("GROQ_API_KEY")
)

# ── Test dataset ──
test_data = [
    {
        "question": "What is my portfolio dividend yield?",
        "ground_truth": "2.22%"
    },
    {
        "question": "What is my portfolio PE ratio?",
        "ground_truth": "25.7"
    },
    {
        "question": "What percentage of my portfolio is large cap?",
        "ground_truth": "63.1%"
    },
    {
        "question": "What is my portfolio profit growth?",
        "ground_truth": "6.1%"
    },
    {
        "question": "What is my portfolio PEG ratio?",
        "ground_truth": "1.16"
    },
]

def evaluate_faithfulness(question, answer, context, llm):
    prompt = f"""Given the context and answer below, is the answer faithful to the context?
    Reply with ONLY a single number: 1 if faithful, 0 if not faithful. Nothing else.

    Context: {context}
    Answer: {answer}
    Reply with only 0 or 1:"""
    response = llm.invoke(prompt)
    try:
        # extract first number found in response
        import re
        numbers = re.findall(r'\d+\.?\d*', response.content.strip())
        return float(numbers[0]) if numbers else 0.0
    except:
        return 0.0

def evaluate_relevancy(question, answer, llm):
    prompt = f"""Is this answer relevant to the question?
    Reply with ONLY a single number: 1 if relevant, 0 if not relevant.

    Question: {question}
    Answer: {answer}
    Reply with only 0 or 1:"""
    response = llm.invoke(prompt)
    try:
        import re
        numbers = re.findall(r'\d+\.?\d*', response.content.strip())
        return float(numbers[0]) if numbers else 0.0
    except:
        return 0.0

def evaluate_correctness(answer, ground_truth, llm):
    prompt = f"""Does the answer contain the correct information from ground truth?
    Reply with ONLY a single number: 1 if correct, 0 if incorrect.

    Ground truth: {ground_truth}
    Answer: {answer}
    Reply with only 0 or 1:"""
    response = llm.invoke(prompt)
    try:
        import re
        numbers = re.findall(r'\d+\.?\d*', response.content.strip())
        return float(numbers[0]) if numbers else 0.0
    except:
        return 0.0

def run_evaluation():
    print("Loading FinPilot...")
    dense_index, sparse_index, bm25, embeddings, reranker = load_retriever()

    results = []
    print("\nRunning evaluation...\n")

    for item in test_data:
        question = item["question"]
        ground_truth = item["ground_truth"]

        # get answer and context
        chunks = retrieve(question, dense_index, sparse_index, bm25, embeddings, reranker)
        context = format_context(chunks)
        answer = chat(question, dense_index, sparse_index, bm25, embeddings, reranker)

        # evaluate
        faithfulness = evaluate_faithfulness(question, answer, context, llm)
        relevancy = evaluate_relevancy(question, answer, llm)
        correctness = evaluate_correctness(answer, ground_truth, llm)

        results.append({
            "question": question,
            "faithfulness": faithfulness,
            "relevancy": relevancy,
            "correctness": correctness
        })

        print(f"Q: {question}")
        print(f"   Faithfulness: {faithfulness:.2f} | Relevancy: {relevancy:.2f} | Correctness: {correctness:.2f}")
        print()

    # ── Print summary ──
    print("="*50)
    print("FinPilot Evaluation Summary")
    print("="*50)
    avg_faith = sum(r["faithfulness"] for r in results) / len(results)
    avg_rel   = sum(r["relevancy"]    for r in results) / len(results)
    avg_corr  = sum(r["correctness"]  for r in results) / len(results)
    print(f"Avg Faithfulness:  {avg_faith:.2f}")
    print(f"Avg Relevancy:     {avg_rel:.2f}")
    print(f"Avg Correctness:   {avg_corr:.2f}")
    print(f"Overall Score:     {(avg_faith + avg_rel + avg_corr) / 3:.2f}")
    print("="*50)

if __name__ == "__main__":
    run_evaluation()