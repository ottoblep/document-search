qna_model_file = "./models/llama-2-7b-chat.Q4_K_M.gguf"
localai_address = "http://localhost:8080"

from langchain.document_loaders import WikipediaLoader 
from langchain.embeddings import LocalAIEmbeddings
from langchain.llms import LlamaCpp
from langchain.retrievers import WikipediaRetriever
from langchain.chains import RetrievalQA

print("Loading LLM")
llm = LlamaCpp(model_path=qna_model_file, n_ctx=4096)

print("Loading QAChain")
qa_chain = RetrievalQA.from_chain_type(llm,retriever=WikipediaRetriever())

while True:
    question = input("Ask wikipedia a question: ")
    print("Generating answer")
    print(qa_chain({"query": question}))
