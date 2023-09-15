from llm import *
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import WikipediaQueryRun
from langchain.agents import initialize_agent, AgentType, Tool

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

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    max_iterations = 1, early_stopping_method = "generate", verbose=True, handle_parsing_errors=True)

while True:
    question = input("Ask a question: ")
    agent.run(question)
