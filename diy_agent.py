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
    You are a Chatbot Agent. Answer the following questions as best you can. You have access to the following tools:

    Search: useful for gaining recent and fast changing information.
    Wiki: useful for researching static facts.

    Use the following format:
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be Search or Wiki 
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    <</SYS>>

    [INST] {question} [/INST]
    """

def parse_response(response):
    start = "Action:"
    end = "Action Input:"
    action = (response.split(start))[1].split(end)[0][:-1]
    action_input = (response.split(end))[1][:-1]
    return action, action_input

def take_action(action, action_input):
    if "Search" in action: 
        return search.run(action_input)
    if "Wiki" in action: 
        return wiki.run(action_input)

while True:
    question = input("Ask a question: ")
    prompt = initial_prompt(question)
    print(prompt)
    response = llm(prompt, stop=["Observation: "])
    print(response)
    action, action_input = parse_response(response)
    action_result = take_action(action, action_input)
    print(action_result)
    followup_prompt = prompt + response + "Observation: " + action_result + "\n"
    followup_response = llm(followup_prompt)
    print(followup_response)