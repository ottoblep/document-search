from llm import *
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import WikipediaQueryRun
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.agents.tools import Tool

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

planner = load_chat_planner(model)

executor = load_agent_executor(llm, tools, verbose=True)

agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

agent.run("Who holds the german championship in disc throwing?")