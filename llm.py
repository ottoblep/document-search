from langchain.llms import LlamaCpp
from langchain.embeddings import LocalAIEmbeddings

qna_model_file = "./models/llama-2-7b-chat.Q4_K_M.gguf"

llm = LlamaCpp(model_path=qna_model_file, n_ctx=4096, n_threads=12, temperature=0.2, echo=True)

localai_address = "http://localhost:8080"

embeddings = LocalAIEmbeddings(openai_api_base=localai_address, model="bert-embeddings", openai_api_key="mock-key")
