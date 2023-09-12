import os
from llm import *
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.embeddings import LocalAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import LlamaCpp

faiss_database = 'vectorstore'   #keep multiple files (.txt, .pdf) in data folder.

print("Loading Vectorstore")
vectorstore = FAISS.load_local(folder_path=faiss_database, embeddings=embeddings)

qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectorstore.as_retriever())

while True:
  question = input("Ask a question: ")
  print("Extracting Context:")
  docs = vectorstore.similarity_search(question, k=1)
  print(docs)
  
  print("Generating answer")
  
  print(qa_chain({"query": question}))