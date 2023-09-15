from llm import *
from langchain.llms import LlamaCpp

qna_model_file = "./models/llama-2-7b-chat.Q4_K_M.gguf"
llm = LlamaCpp(model_path=qna_model_file, n_ctx=4096, n_threads=12, temperature=0.2, echo=True)

def initial_prompt():
    return f"""
    <<SYS>>
    You are an interactive assistant. Determine what the user wants to do.
    Your task is to either do Nothing or use a tool.
    Only use a tool if it is explicitly requested.
    If the intention of the user is unclear do Nothing.
    You have access to the following tools:

    Nothing: useful if the instruction is unclear 
    Jump: useful for moving into a higher place 
    Websearch: useful for looking up recent or fast changing information 
    Wikipedia: useful for looking up static knowledge
    Amazon: useful to buy things.
    Spotify: useful for listening to music.
    Doordash: useful to buy groceries.
    Uber: useful for going somewhere.
    Lieferando: useful for ordering food.

    Use the following format:

    Thought:  you should always think about what to do
    Response: a response to give to the user
    Action: the action to take, should be either Nothing, Jump, Websearch, Wikipedia, Amazon, Spotify, Doordash, Uber or Lieferando.
    <</SYS>>

    """

def add_instruction(instruction, prompt):
    return f"{prompt} [INST] {instruction} [/INST] "

while True:
    prompt = initial_prompt()
    while True:
        question = input("LLM>>")
        prompt = add_instruction(question, prompt)
        if "reset" in question: break
        print(prompt)
        response = llm(prompt)
        prompt += response
        print(response)