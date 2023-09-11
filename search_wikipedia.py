qna_model = "llama2-chat"
localai_address = "http://localhost:8080"

from langchain.document_loaders import WikipediaLoader 
from langchain.embeddings import LocalAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import WikipediaRetriever
from langchain.chains import RetrievalQA

print("Loading LLM")
llm = ChatOpenAI(model_name=qna_model, openai_api_base=localai_address, openai_api_key="mock-key", streaming=True, temperature=0.2, max_tokens=200)

print("Loading QAChain")
qa_chain = RetrievalQA.from_chain_type(llm,retriever=WikipediaRetriever())

while True:
    question = input("Ask wikipedia a question: ")
    print("Generating answer")
    print(qa_chain({"query": question}))
