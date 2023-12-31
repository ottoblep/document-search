from llm import *
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.utilities import WikipediaAPIWrapper
from langchain.llms import LlamaCpp

search = DuckDuckGoSearchAPIWrapper()
wiki = WikipediaAPIWrapper(top_k_result=2, doc_content_chars_max=512)

def initial_prompt():
    return f"""
    <<SYS>>
    You are a Chatbot Agent. Answer the following questions as best you can. You have access to the following tools:

    Web: useful for looking up recent and fast changing information.
    Wiki: useful for looking up facts and knowledge that doesn't change often.
    None: useful if you already know the answer or the answer is obvious

    Use the following format:
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be Web or Wiki or None
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    <</SYS>>

    """

def parse_response(response):
    start = "Action:"
    end = "Action Input:"
    try: 
        action = (response.split(start))[1].split(end)[0].strip('\n')
        action_input = (response.split(end))[1].strip('\n')
    except:
        return "", ""
    return action, action_input

def take_action(action, action_input):
    if "Web" in action: 
        return search.run(action_input)
    if "Wiki" in action: 
        return wiki.run(action_input)
    return ""

def add_instruction(instruction, prompt):
    return f"{prompt} [INST] {instruction} [/INST] "

while True:
    prompt = initial_prompt()
    while True:
        question = input("LLM>>")
        if "reset" in question: break
        prompt = add_instruction(question, prompt)
        print(prompt)
        response = llm(prompt, stop=["Observation: "])
        print(response)
        action, action_input = parse_response(response)
        action_result = take_action(action, action_input)
        print(action_result)
        followup_prompt = prompt + response + "Observation: " + action_result + "\n"
        followup_response = llm(followup_prompt)
        print(followup_response)
        prompt = followup_prompt + followup_response