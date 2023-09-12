from llm import *

from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import WikipediaQueryRun

search = DuckDuckGoSearchAPIWrapper()
wiki = WikipediaAPIWrapper(top_k_result=2, doc_content_chars_max=512)

tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about fast changing or current events. You should ask targeted questions"
    ),
    Tool(
        name = "Wiki",
        func=wiki.run,
        description="useful for looking up static and long standing facts and information about things."
    ),
]

memory = ConversationBufferMemory(memory_key="chat_history")
agent_chain = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)

while True:
    question = input("Ask>> ")
    agent_chain.run(input=question)