from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent

# for memory categorisation we can use Pydantic 
from pydantic import BaseModel, Field
from typing import Literal

class MemoryModel:
    should_remember: bool
    type: Literal[
        "None",
        "PROFILE",
        "PREFERENCE",
        "GOAL",
        "FACT",
        "TEMPORARY"
    ]
    memory: str
    importance: float = Field(
        ge = 0,
        le = 1
    )

memory_model = model.with_structured_output(MemoryModel)

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

def save_db(memory):
    return "saved"

messages = []

user_query = input("User Query: ")

messages.append({
    "role": "user",
    "content": user_query
})

memory = memory_model(user_query)

if (
    memory.should_remember
    and memory.type in ["PROFILE", "PREFERENCE", "FACT", "GOAL"]
    and memory.importance >= 0.7
):
    save_db(memory.memory)


model = init_chat_model(
    "claude-4-8",
)

tools = [
    calculate, 
    get_weather,
    search_db
]

model_with_tools = model.bind_tools(tools)

response = model.invoke(messages)   # without tools access LLM call
response = model_with_tools.invoke(messages)    # with tools access LLM call

# In scratch
# LLM
#  ↓
# check tool_calls
#  ↓
# execute
#  ↓
# append result
#  ↓
# LLM again
#  ↓
# check again
#  ↓
# execute again
# In scratch we used while True to handle loop using LangChain
# it is handled

# The agent handles the loop.
agent = create_agent(
    model = model,
    tools = tools
)
result = agent.invoke(messages)