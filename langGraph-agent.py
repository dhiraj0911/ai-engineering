from typing import TypedDict, Annotated

from langchain.chat_models import init_chat_model
from langchain.tools import tool

from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.prebuilt import ToolNode

@tool
def get_weather(city: str):
    """Get the current weather for a city"""
    # Make external API call or MCP server call
    return {
        "city": city,
        "temperature": 40,
        "unit": "C"
    }

@tool
def search_db(id: str):
    """Get the customer related information from database"""
    # make external api call
    return {
        "id": id,
        "data": "temp data"
    }

@tool
def calculate(a: float, b: float, ops: str):
    """Perform a mathematical calculation"""
    if ops == "+":
        return a + b
    elif ops == "-":
        return a - b
    elif ops == "*":
        return a * b
    elif ops == "/":
        return a / b
    else:
        raise ValueError(f"Unsupported operation: {ops}")


model = init_chat_model(
    "claude-4-8",
)

tools = [
    calculate, 
    get_weather,
    search_db
]

model_with_tools = model.bind_tools(tools)

# state-> The information flowing through your graph 
# and every node can read/upate state
class agentState(TypedDict):
    messages: Annotated[list, add_messages]

"""
messages = [
    user message,
    AI message,
    tool result
]
"""    

# Edges-> Define where to go next

# node-> A python function that performs one step of workflow
# create LLM Node
def call_llm(state: agentState):
    response = model_with_tools.invoke(state['messages'])

    return {
        "message": [response]
    }

# Tool Node
# Instead of manually doing:
# for tool_call in response.tool_calls:
#   ...

tool_node = ToolNode(tools)


def should_continue(state: agentState):
    last_message = state['messages'][-1]

    if last_message.tool_call:
        return "tool"

    return "end"


# Now build the graph
builder = StateGraph(agentState)

# add nodes
builder.add_node("call", call_llm)
builder.add_node("tools", tool_node)

# form edges
builder.add_edge(START, "call")
builder.add_conditional_edges(
    "call",
    should_continue,
    {
        "tool": "tools",
        "end": END
    }
)

builder.add_edge("tools", "call")


graph = builder.compile()

result = graph.invoke({
    "messsage": {
        "role": "user",
        "query": "could you explain me how is flow of your system?"
    }
})

