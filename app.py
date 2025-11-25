from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, RemoveMessage
from typing import Literal
from IPython.display import Image, display
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.document_loaders import WebBaseLoader, AsyncChromiumLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_text_splitters.character import CharacterTextSplitter
import asyncio
import os
import copy
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"


# import requests


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
    summary:str
    
# def filter_messages(state: MessagesState):
#     delele_messages = [RemoveMessage(id=msg.id) for msg in state['messages'][:-2]]


def call_model(state: State):

    summary = state.get("summary", "")

    if summary:
        system_message = f"You are a helpful assistant! Here is the conversation summary so far: {summary}"
        
        messages = [SystemMessage(content=system_message)] + state["messages"]

    else:
        messages = state["messages"]


    response = chat.invoke(messages)
    return {"messages": state["messages"] + [response]}


def summarize_conversation(state: State):
    print("\n=== SUMMARIZING CONVERSATION ===\n")
    summary = state.get("summary", "")

    if summary:
        
        # A summary already exists
        summary_message = (
            f"This is a summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
        
    else:
        summary_message = "Create a summary of the conversation above:"

    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = chat.invoke(messages)

    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    new_messages = state["messages"][-2:]
    return {"summary": response.content, "messages": delete_messages + new_messages}

def should_summarize(state: State) -> Literal["summarize",END]:
    """Return the next node to execute."""

    messages = state["messages"]

    if len(messages) > 6:
        return "summarize"
    else:
        return END


async def load_website(urls: list[str]) -> list[str]:
    loader = AsyncChromiumLoader(urls, user_agent="MyAppUserAgent")
    docs = await loader.aload()
    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)
    return docs_transformed[0].page_content

builder = StateGraph(State)
builder.add_node("model", call_model)
builder.add_node("summarize", summarize_conversation)


builder.add_edge(START, "model")
builder.add_conditional_edges("model", should_summarize)
builder.add_edge("summarize", END)
builder.add_edge("model", END)



graph = builder.compile()
memory = MemorySaver()
react_graph_memory = builder.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "1"}}


result = asyncio.run(load_website(["https://kemendikdasmen.go.id/berita/14190-kemendikdasmen-prioritaskan-pengembangan-konten-bidang-kejur"]))
# print(result)
# print(len(result))
#resul 9519
string_cleaned = copy.deepcopy(result)
string_cleaned = string_cleaned.strip().replace("\n"," ").replace("#"," "). replace("*"," ")
# print(string_cut)
# print(len(string_cut))
char_splitter = CharacterTextSplitter(
    separator=" ",
    chunk_size=2000,
    chunk_overlap=200,
)

pages_char_split = char_splitter.split_text(string_cleaned)
print(pages_char_split[-2])







