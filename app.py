from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, RemoveMessage
from typing import Literal
from IPython.display import Image, display
from langgraph.checkpoint.memory import MemorySaver


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
chat = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


class State(MessagesState):
    pass


def call_model(state: State):
    response = chat.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}

builder = StateGraph(State)
builder.add_node("model", call_model)

builder.add_edge(START, "model")
builder.add_edge("model", END)

graph = builder.compile()
memory = MemorySaver()
react_graph_memory = builder.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "1"}}

messages= [
    SystemMessage(content="You are a helpful assistant! Your name is Bob."),
    HumanMessage(content="What is your name?"),
]
messages_no_memories = graph.invoke({"messages": messages})
messages_no_memories = graph.invoke(
    {"messages": [HumanMessage(content="What is your name again?")]}
)

messages_w_memories = react_graph_memory.invoke({"messages": messages}, config)
messages_w_memories = react_graph_memory.invoke(
    {"messages": [HumanMessage(content="What is your name again?")]},
    config
)


print("\n\n===== WITHOUT MEMORIES =====")
for i in messages_no_memories["messages"]:
    i.pretty_print()                        
print("\n\n===== WITH MEMORIES =====")
for i in messages_w_memories["messages"]:
    i.pretty_print()    

# print(result["messages"][-1].content)
