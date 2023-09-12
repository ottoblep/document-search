import os
from llm import *
from langchain.document_loaders import WikipediaLoader, DirectoryLoader
from langchain.retrievers import WikipediaRetriever
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS

faiss_database = "vectorstore"

print("Loading Vectorstore")
vectorstore = FAISS.load_local(folder_path=faiss_database, embeddings=embeddings)

qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectorstore.as_retriever())

print("Loading Retriever")
wiki_retriever = WikipediaRetriever(doc_content_chars_max = 1000)

while True:
  question = input("Ask a question: ")
  print("Adding relevant wikipedia articles to search:")
  wiki_docs = wiki_retriever.get_relevant_documents(query = question)
  vectorstore.add_documents(wiki_docs, embeddings = embeddings)
  print("Extracting Context:")
  docs = vectorstore.similarity_search(question, k=1)
  print(docs)
  print("Generating answer")
  print(qa_chain({"query": question}))