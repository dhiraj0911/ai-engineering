def get_weather(city):
    # Make external API call or MCP server call
    return {
        "city": city,
        "temperature": 40,
        "unit": "C"
    }

def search_db(id):
    # make external api call
    return {
        "id": id,
        "data": "temp data"
    }

def save_db(memory):
    return "saved"

def calculate(a, b, ops):
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


def decide_categorie(text):
    prompt = """
        Analyze above message.
        Determine whether it contains information worth
        remembering across future conversations.

        Categories:
        - NONE
        - PROFILE
        - PREFERENCE
        - GOAL
        - FACT
        - TEMPORARY

        Return following structured JSON.
        {
            "should_remember": boolean,
            "type": One categories,
            "memory": Result to store,
            "importance": from 0 to 1 (higher number more importance)
        }
    """

    response = client.call(
        model="haiku",
        messages=text + prompt
    )
    return response

def saveMemory(result):
    if result["type"] == "PROFILE" or result["type"] == "PREFERENCE":
        save_db(result["memory"])
    elif result["type"] == "FACT" or result["type"] == "GOAL" or result["type"] == "TEMPORARY":
        messages.append({
            "preference": result["type"],
            "memory": result["memory"],
            "importance": result["importance"]
        })
    else:
        return None

def execute_tool(name, args):
    if name == "get_weather":
        return get_weather(args["city"])

    elif name == "calculate":
        return calculate(
            args["a"],
            args["b"],
            args["ops"]
        )

    elif name == "search_db":
        return search_db(args["id"])

    else:
        raise ValueError(f"Tool not available: {name}")


# ---------------------------------------
# Tool metadata given to the LLM
# ---------------------------------------

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Name of the city"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_db",
            "description": "Get the customer related information from database",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Unique id of user"
                    }
                },
                "required": ["id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform a mathematical calculation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number"
                    },
                    "b": {
                        "type": "number"
                    },
                    "ops": {
                        "type": "string",
                        "enum": ["+", "-", "*", "/"]
                    }
                },
                "required": ["a", "b", "ops"]
            }
        }
    }
]


# ---------------------------------------
# Conversation
# ---------------------------------------

messages = []

user_query = input("User Query: ")

messages.append({
    "role": "user",
    "content": user_query
})


# ---------------------------------------
# First LLM call
# ---------------------------------------

memory = decide_categorie(user_query)
saveMemory(memory)

response = client.call(
    model=model,
    messages=messages,
    tools=tools
)

response_message = response.choices[0].message


# ---------------------------------------
# Check whether LLM wants a tool
# ---------------------------------------

if response_message.tool_calls:

    # Add LLM's response containing tool calls
    messages.append(response_message)

    # Execute every requested tool
    for tool_call in response_message.tool_calls:

        tool_name = tool_call.function.name

        tool_args = json.loads(
            tool_call.function.arguments
        )

        result = execute_tool(
            tool_name,
            tool_args
        )

        # Add tool result to conversation
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result)
        })


    # ---------------------------------------
    # Second LLM call
    # ---------------------------------------

    final_response = client.call(
        model=model,
        messages=messages,
        tools=tools
    )

    print(
        "Final result:",
        final_response.choices[0].message.content
    )

else:

    # No tool required
    print(
        "Final result:",
        response_message.content
    )