def get_weather(city):
    # make external api call or MCP server

    return {
        city: 40
    }

def calculate(a, b, ops): 
    if ops == '+':
        return a + b
    elif ops == '-':
        return a - b
    elif ops == '*':
        return a * b
    else:
        return a / b

def execute_tool(name, args):
    if name == 'get_weather':
        return get_weather(args['city'])
    elif name == 'calculate':
        return calculate(args['a'], args['b'], args['ops'])
    return "Not Avaiable"

tools = {
    # all tool metadata contains detailed
    # about tool/function avaiable
    """name, description, arg, return type"""
}

messages = [];

user_query = input('User Query: ')

messages.append({
    "role": "user",
    "message": user_query
})

response = "test"
# llm call
# response = client.create({
#     model='gpt-astra',
#     messages=messages,
#     tools=tools
# })


#now response must have a message from LLM if there any tool required to acompolised task

response_message = response.choices[0].message

if response_message.tool_call:
    messages.append(response_message)
    for tool in response_message.tool_call:
        toolName = response_message.tool_call.name
        toolArgs = response_message.tool_call.args
        result = execute_tool(toolName, toolArgs)


        messages.append({
            "role": tool,
            "tool_name": toolName,
            "tool_arg": toolArgs,
            "result": result
        })



    # Ask final call LLM again

    final_result = client.call({
        model,
        messages,
        tools
    })

    print("Final result, ", final_result)



