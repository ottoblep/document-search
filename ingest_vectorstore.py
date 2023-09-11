import os.path
from langchain.document_loaders import WikipediaLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import LocalAIEmbeddings

directory = "src"
faiss_database = "vectorstore"
localai_address = "http://localhost:8080"

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

if os.path.isfile(faiss_database+"/index.faiss"):
    print("Adding to vectorstore")
    vectorstore = FAISS.load_local(faiss_database, embeddings=embeddings)
    vectorstore.add_documents(all_splits)
    vectorstore.save_local(faiss_database)
else:
    print("Creating new vectorstore")
    vectorstore = FAISS.from_documents(documents=all_splits, embedding=embeddings)
    vectorstore.save_local(faiss_database)



