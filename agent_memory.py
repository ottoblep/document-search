from llm import *
import os
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.utilities import WikipediaAPIWrapper
from langchain.llms import LlamaCpp
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

faiss_database = 'vectorstore_agent'

if os.path.isfile(faiss_database+"/index.faiss"):
    vectorstore = FAISS.load_local(folder_path=faiss_database, embeddings=embeddings)
else:
    vectorstore = FAISS.from_texts(texts=[""], embedding=embeddings)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=20)
search = DuckDuckGoSearchAPIWrapper()
wiki = WikipediaAPIWrapper(top_k_result=2, doc_content_chars_max=512)

def initial_prompt():
    return f"""
<<SYS>>
You are a Chatbot Agent. Answer the following questions as best you can. You have access to the following tools:

Duck: useful for looking up recent and fast changing information.
Wiki: useful for looking up facts and knowledge that doesn't change often.
None: useful if you already have enough information or the answer is obvious

Use the following format:
Question: the input question you must answer
Memory: information from past interactions
Thought: you should always think about what to do
Action: the action to take, should be Duck or Wiki or None
Action Input: the input query for the action
Observation: the result of the action
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
    if "Duck" in action: 
        return search.run(action_input)
    if "Wiki" in action: 
        return wiki.run(action_input)
    return ""

def add_instruction(instruction, prompt):
    return f"{prompt}\n[INST] {instruction} [/INST]\n"

def docs_to_text(document_list):
    result = ""
    for doc in document_list:
        result += doc.page_content
    return result

while True:
    prompt = initial_prompt()
    iteration = 1
    while True:
        # Format prompt for question
        question = input("LLM>> ")
        if "reset" in question: break
        prompt = add_instruction(question, prompt) + "\n"
        prompt += "Question: " + question + "\n"

        # Only look up memory once per conversation
        if iteration == 1:
            prompt += "Memory: " + docs_to_text(vectorstore.similarity_search(question, k=3)) + "\n"
        prompt += "Thought: "
        print(prompt)

        # Choose Action
        response = llm(prompt, stop=["Observation: "])
        action, action_input = parse_response(response)
        print(response)

        # Take action and save results in memory
        action_result = take_action(action, action_input)
        if action_result != "":
            new_texts = text_splitter.split_text(action_result)
            vectorstore.add_texts(texts=new_texts, embeddings=embeddings)
        else: observation = "None"
        print("Action Result: ", action_result)

        # Summarize Results
        prompt += response + "Observation: " + action_result + "\n"
        response = llm(prompt)
        print(response)
        prompt += response
        vectorstore.save_local(faiss_database)
        iteration += 1