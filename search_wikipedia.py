from llm import *
from langchain.document_loaders import WikipediaLoader 
from langchain.retrievers import WikipediaRetriever
from langchain.chains import RetrievalQA

print("Loading QAChain")
qa_chain = RetrievalQA.from_chain_type(llm,retriever=WikipediaRetriever())

while True:
    question = input("Ask wikipedia a question: ")
    print("Generating answer")
    print(qa_chain({"query": question}))
