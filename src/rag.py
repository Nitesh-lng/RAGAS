from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

from src.config import LLM_MODEL, INDEX_PATH, TOP_K
from src.vector_store import VectorStoreBuilder

load_dotenv()

class SimpleRag():

    def __init__(self):

        self.vector_store= VectorStoreBuilder().load(INDEX_PATH)
        self.llm= ChatGroq(model=LLM_MODEL)
        self.prompt = ChatPromptTemplate.from_template(
            """Answer the question using ONLY the context below.
            If the context does not contain the answer, say "I don't have enough information."

            Context:
            {context}

            Question: {question}

            Answer:"""
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    def query(self,query):
        docs= self.vector_store.similarity_search(query,k=3)
        context= [doc.page_content for doc in docs]

        context_text= '\n\n'.join(context)
        answer= self.chain.invoke({
            'context':context_text,
            'question': query
        })

        return {
            'answer': answer,
            'contexts': context
        }
    