from llm import *
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.utilities import WikipediaAPIWrapper
from langchain.llms import LlamaCpp

qna_model_file = "./models/llama-2-7b-chat.Q4_K_M.gguf"
llm = LlamaCpp(model_path=qna_model_file, n_ctx=4096, n_threads=12, temperature=0.2, echo=True)

search = DuckDuckGoSearchAPIWrapper()
wiki = WikipediaAPIWrapper(top_k_result=2, doc_content_chars_max=512)

def initial_prompt(question):
    return f"""
    <<SYS>>
    You are an interactive Door opener. Determine if the user wants to open the door.
    Your task is to either do Nothing or operate the door if requested.
    Only open or close the door if it is explicitly requested.
    If the intention of the user is unclear do Nothing.
    You have access to the following tools:

    Nothing: useful if the instruction is unrelated to the door 
    Open: useful if the user wants to open the door 
    Close: useful if the user wants to close the door 

    Use the following format:

    Thought:  you should always think about what to do
    Response: a response to give to the user
    Action: the action to take, should be Open or Close or Nothing
    <</SYS>>

    [INST] {question} [/INST]
    """

while True:
    question = input("LLM>>")
    prompt = initial_prompt(question)
    print(prompt)
    response = llm(prompt)
    print(response)