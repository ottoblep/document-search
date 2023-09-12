from langchain.llms import LlamaCpp
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.retrievers import WikipediaRetriever
from langchain.agents import initialize_agent, AgentType, Tool

qna_model_file = "./models/llama-2-7b-chat.Q4_K_M.gguf"

llm = LlamaCpp(model_path=qna_model_file, n_ctx=4096)
search = DuckDuckGoSearchAPIWrapper()
wiki = WikipediaRetriever(doc_content_chars_max = 1000)

tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions"
    ),
    Tool(
        name = "Wiki",
        func=wiki.load,
        description="useful for looking up well known facts and information about things."
    ),
]

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION , verbose=True)

agent.run("What is the current stock price of NVIDIA?")