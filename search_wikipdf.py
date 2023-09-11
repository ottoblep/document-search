directory = 'src'   #keep multiple files (.txt, .pdf) in data folder.
qna_model = "llama2-chat"
localai_address = "http://localhost:8080"

import os
from langchain.document_loaders import WikipediaLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import LocalAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import WikipediaRetriever
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS

def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents

print("Loading Documents")
documents = load_docs(directory)

def split_docs(documents, chunk_size=1000, chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

print("Splitting Documents")
all_splits = split_docs(documents)
embeddings = LocalAIEmbeddings(openai_api_base=localai_address, model="bert-embeddings", openai_api_key="mock-key")

print("Loading Vectorstore")
vectorstore = FAISS.from_documents(documents=all_splits, embedding=embeddings)

print("Loading LLM")
llm = ChatOpenAI(model_name=qna_model, openai_api_base=localai_address, openai_api_key="mock-key", streaming=True, temperature=0.2, max_tokens=200)
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