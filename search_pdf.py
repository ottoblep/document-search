import os
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.embeddings import LocalAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import LlamaCpp

qna_model_file = "./models/llama-2-7b-chat.Q4_K_M.gguf"
localai_address = "http://localhost:8080"
faiss_database = 'vectorstore'   #keep multiple files (.txt, .pdf) in data folder.

embeddings = LocalAIEmbeddings(openai_api_base=localai_address, model="bert-embeddings", openai_api_key="mock-key")

print("Loading Vectorstore")
vectorstore = FAISS.load_local(folder_path=faiss_database, embeddings=embeddings)

print("Loading LLM")
llm = LlamaCpp(model_path=qna_model_file, n_ctx=4096)
qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectorstore.as_retriever())

while True:
  question = input("Ask a question: ")
  print("Extracting Context:")
  docs = vectorstore.similarity_search(question, k=1)
  print(docs)
  
  print("Generating answer")
  
  print(qa_chain({"query": question}))