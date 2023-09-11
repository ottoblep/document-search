import os
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.embeddings import LocalAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

qna_model = "llama2-chat"
localai_address = "http://localhost:8080"
faiss_database = 'vectorstore'   #keep multiple files (.txt, .pdf) in data folder.

embeddings = LocalAIEmbeddings(openai_api_base=localai_address, model="bert-embeddings", openai_api_key="mock-key")

print("Loading Vectorstore")
vectorstore = FAISS.load_local(folder_path=faiss_database, embeddings=embeddings)

print("Loading LLM")
llm = ChatOpenAI(model_name=qna_model, openai_api_base=localai_address, openai_api_key="mock-key", streaming=True, temperature=0.2, max_tokens=200)
qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectorstore.as_retriever())

while True:
  question = input("Ask a question: ")
  print("Extracting Context:")
  docs = vectorstore.similarity_search(question, k=1)
  print(docs)
  
  print("Generating answer")
  
  print(qa_chain({"query": question}))