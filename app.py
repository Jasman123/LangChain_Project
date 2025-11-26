from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, RemoveMessage
from typing import Literal
from IPython.display import Image, display

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


def chat_model(state: State) -> State:
    response = chat.invoke(state['messages'])
    return {"messages": state['messages'] + [response]}


builder = StateGraph(State)
builder.add_node("chat_model", chat_model)

builder.add_edge(START, "chat_model")
builder.add_edge("chat_model", END)

graph = builder.compile()


messages = [
    SystemMessage(content="You are a helpful assistant! Your name is Bob."),
]

if __name__ == "__main__":
    print("Chatbot ready...\n")
    while True:
        user_input= input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chat...")
            break
        
        messages.append(HumanMessage(content=user_input))

        response = graph.invoke({"messages": messages})
        messages = response['messages'] 
        for msg in response['messages']:
            # print(f"{msg.type}: {msg.content}")
             msg.pretty_print()  
        print("\n================New Chat Turn===================\n")



