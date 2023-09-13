import os
from llm import *
from langchain.document_loaders import WikipediaLoader, DirectoryLoader
from langchain.retrievers import WikipediaRetriever
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

faiss_database = "vectorstore"

print("Loading Vectorstore")
vectorstore = FAISS.load_local(folder_path=faiss_database, embeddings=embeddings)

qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectorstore.as_retriever())

print("Loading Retriever")
wiki_retriever = WikipediaRetriever()

while True:
  question = input("Ask a question: ")
  print("Adding relevant wikipedia articles to search:")
  wiki_docs = wiki_retriever.get_relevant_documents(query = question,)
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
  wiki_docs = text_splitter.split_documents(wiki_docs)
  vectorstore.add_documents(wiki_docs, embeddings = embeddings)
  print("Extracting Context:")
  docs = vectorstore.similarity_search(question, k = 3)
  print(docs)
  print("Generating answer")
  print(qa_chain({"query": question}))