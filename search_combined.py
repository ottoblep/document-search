import os
from llm import *
from langchain.document_loaders import WikipediaLoader, DirectoryLoader
from langchain.retrievers import WikipediaRetriever
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import LLMChain, PromptTemplate
from langchain.utilities import DuckDuckGoSearchAPIWrapper

faiss_database = "vectorstore"

print("Loading Vectorstore")
vectorstore = FAISS.load_local(folder_path=faiss_database, embeddings=embeddings)

qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectorstore.as_retriever())
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
wiki_retriever = WikipediaRetriever()
search_wrapper = DuckDuckGoSearchAPIWrapper()

while True:
  question = input("Ask a question: ")
  print("Retrieving wikipedia")
  wiki_docs = wiki_retriever.get_relevant_documents(query = question)
  new_docs = text_splitter.split_documents(wiki_docs)
  vectorstore.add_documents(new_docs, embeddings = embeddings)
  print("Retrieving search results")
  search_texts = search_wrapper.run(query=question)
  new_texts = text_splitter.split_text(search_texts)
  print(new_texts)
  vectorstore.add_texts(texts=new_texts, embeddings=embeddings)
  print("Extracting Context:")
  docs = vectorstore.similarity_search(question, k=3)
  print(docs)
  print("Generating answer")
  print(qa_chain({"query": question}))