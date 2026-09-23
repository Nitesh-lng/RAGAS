"""
RAGAS-free RAG evaluation.

Instead of the ragas library (which has heavy dependency conflicts),
we implement the core evaluation metrics ourselves using an LLM as a judge.
This is exactly what RAGAS does internally — we just make it transparent.

Metrics implemented:
  1. Faithfulness      -> is the answer grounded in the retrieved context?
  2. Answer Relevancy  -> does the answer address the question?
  3. Context Precision -> are the retrieved chunks relevant to the question?
  4. Context Recall    -> does the context cover the ground-truth answer?
"""

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.rag import SimpleRag
from src.config import LLM_MODEL

# Golden dataset — questions + ideal (ground-truth) answers.
# ground_truth is written by a human, based on what the PDF actually says.

test_data = [
    {
        "question": "What are the two main components of a RAG system?",
        "ground_truth": "A RAG system has two components: a retriever that retrieves information from external sources, and a generator that generates a response based on the retrieved information.",
    },
    {
        "question": "What does the success of a RAG system depend on?",
        "ground_truth": "The success of a RAG system depends on the quality of its retriever.",
    },
    {
        "question": "What is the purpose of the retriever in RAG?",
        "ground_truth": "The retriever fetches relevant information from external memory sources to provide context for generation.",
    },
]

# Judge LLM — this grades the outputs. (RAGAS uses an LLM the same way.)
judge_llm = ChatGroq(model=LLM_MODEL, temperature=0)


def _to_score(text):
    """LLM returns text like '0.8' — safely turn it into a float in [0, 1]."""
    try:
        value = float(text.strip().split()[0])
        return max(0.0, min(1.0, value))   
    except (ValueError, IndexError):
        return 0.0

# Metric 1 — Faithfulness: answer grounded in the context? (anti-hallucination)
faithfulness_prompt = ChatPromptTemplate.from_template(
    """You are evaluating whether an answer is grounded in the given context.

Score from 0.0 to 1.0:
- 1.0 = every claim in the answer is fully supported by the context
- 0.0 = the answer contains claims not found in the context (hallucination)

Return ONLY the number, nothing else.

Context:
{context}

Answer:
{answer}

Score:"""
)
faithfulness_chain = faithfulness_prompt | judge_llm | StrOutputParser()


def faithfulness_score(answer, contexts):
    context_text = "\n\n".join(contexts)
    result = faithfulness_chain.invoke({"context": context_text, "answer": answer})
    return _to_score(result)

# Metric 2 — Answer Relevancy: does the answer address the question?
relevancy_prompt = ChatPromptTemplate.from_template(
    """You are evaluating whether an answer actually addresses the question.

Score from 0.0 to 1.0:
- 1.0 = the answer directly and completely addresses the question
- 0.0 = the answer is off-topic or does not address the question

Return ONLY the number, nothing else.

Question:
{question}

Answer:
{answer}

Score:"""
)
relevancy_chain = relevancy_prompt | judge_llm | StrOutputParser()


def answer_relevancy_score(question, answer):
    result = relevancy_chain.invoke({"question": question, "answer": answer})
    return _to_score(result)

# Metric 3 — Context Precision: are the retrieved chunks relevant?
precision_prompt = ChatPromptTemplate.from_template(
    """You are evaluating how relevant the retrieved context is to the question.

Score from 0.0 to 1.0:
- 1.0 = all retrieved context is relevant to answering the question
- 0.0 = the retrieved context is mostly irrelevant noise

Return ONLY the number, nothing else.

Question:
{question}

Retrieved context:
{context}

Score:"""
)
precision_chain = precision_prompt | judge_llm | StrOutputParser()


def context_precision_score(question, contexts):
    context_text = "\n\n".join(contexts)
    result = precision_chain.invoke({"question": question, "context": context_text})
    return _to_score(result)

# Metric 4 — Context Recall: does the context cover the ground-truth?
recall_prompt = ChatPromptTemplate.from_template(
    """You are evaluating whether the retrieved context contains the information
needed to produce the ground-truth answer.

Score from 0.0 to 1.0:
- 1.0 = all facts in the ground-truth answer are present in the context
- 0.0 = the context is missing the information in the ground-truth answer

Return ONLY the number, nothing else.

Ground-truth answer:
{ground_truth}

Retrieved context:
{context}

Score:"""
)
recall_chain = recall_prompt | judge_llm | StrOutputParser()


def context_recall_score(ground_truth, contexts):
    context_text = "\n\n".join(contexts)
    result = recall_chain.invoke({"ground_truth": ground_truth, "context": context_text})
    return _to_score(result)

def main():
    rag = SimpleRag()

    faith_scores = []
    relevancy_scores = []
    precision_scores = []
    recall_scores = []

    for item in test_data:
        question = item["question"]
        ground_truth = item["ground_truth"]

      
        result = rag.query(question)
        answer = result["answer"]
        contexts = result["contexts"]

        
        f = faithfulness_score(answer, contexts)
        r = answer_relevancy_score(question, answer)
        p = context_precision_score(question, contexts)
        rec = context_recall_score(ground_truth, contexts)

        faith_scores.append(f)
        relevancy_scores.append(r)
        precision_scores.append(p)
        recall_scores.append(rec)

        print(f"\nQ: {question}")
        print(f"   faithfulness={f:.2f}  answer_relevancy={r:.2f}  "
              f"context_precision={p:.2f}  context_recall={rec:.2f}")

    def avg(xs):
        return sum(xs) / len(xs) if xs else 0.0

    print("\n" + "=" * 45)
    print("       RAG EVALUATION REPORT (averages)")
    print("=" * 45)
    print(f"  Faithfulness      : {avg(faith_scores):.2f}")
    print(f"  Answer Relevancy  : {avg(relevancy_scores):.2f}")
    print(f"  Context Precision : {avg(precision_scores):.2f}")
    print(f"  Context Recall    : {avg(recall_scores):.2f}")
    print("=" * 45)


if __name__ == "__main__":
    main()
