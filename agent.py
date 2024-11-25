from llm import llm
from graph import graph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.tools import Tool
from langchain_community.chat_message_histories import Neo4jChatMessageHistory
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain import hub
from utils import get_session_id
from tools.vector import get_fault_info
from tools.cypher import cypher_qa
from neo4j.exceptions import CypherSyntaxError, ServiceUnavailable

# Create a movie chat chain
# chat_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a nuclear power plant operator expert providing information about incident response."),
#         ("human", "{input}"),
#     ]
# )

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "您是一名核电厂操作专家，负责提供有关异常响应的信息。"),
        ("human", "{input}"),
    ]
)
plant_chat = chat_prompt | llm | StrOutputParser()

# response = plant_chat.invoke({"input": "你是谁？"})
# print(response)
# Create a set of tools
# tools = [
#     Tool.from_function(
#         name="General Chat",
#         description="For general nuclear power plant chat not covered by other tools",
#         func=plant_chat.invoke,
#     ),
#     # Tool.from_function(
#     #     name="Movie Plot Search",  
#     #     description="For when you need to find information about movies based on a plot",
#     #     func=get_movie_plot, 
#     # ),
#      Tool.from_function(
#         name="Movie information",
#         description="Provide information about incident response questions using Cypher",
#         func = cypher_qa
#     )
# ]

tools = [
    # Tool.from_function(
    #     name="General Chat",
    #     description="用于其他工具未涵盖的一般核电厂聊天",
    #     func=plant_chat.invoke,
    # ),
    Tool.from_function(
        name="fault info Search",  
        description="""When the 'Incident response information' tool cannot find information about a fault phenomenon, use this tool to search for fault-related information""",
        func=get_fault_info, 
    ),
    Tool.from_function(
        name="Incident response information",
        description="使用Cypher提供有关异常响应问题的信息",
        func = cypher_qa
    )
]

# Create chat history callback
def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)
    # return ''

# Create the agent

# agent_prompt = PromptTemplate.from_template("""
# You are a nuclear power plant operator expert providing information about incident response.
# Be as helpful as possible and return as much information as possible.
# Do not answer any questions that do not relate to incident response.

# Do not answer any questions using your pre-trained knowledge, only use the information provided in the context.

# TOOLS:
# ------

# You have access to the following tools:

# {tools}

# To use a tool, please use the following format:

# ```
# Thought: Do I need to use a tool? Yes
# Action: the action to take, should be one of [{tool_names}]
# Action Input: the input to the action
# Observation: the result of the action
# ```

# When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

# ```
# Thought: Do I need to use a tool? No
# Final Answer: [your response here]
# ```

# Begin!

# Previous conversation history:
# {chat_history}

# New input: {input}
# {agent_scratchpad}


# """)

agent_prompt = PromptTemplate.from_template("""
你是一位核电站运行专家，提供关于异常响应的信息。
尽可能提供帮助并返回尽可能多的信息。
不要回答任何与异常响应无关的问题。

不要使用你预先训练的知识回答任何问题，只使用上下文中提供的信息,并且使用中文。

工具：
------

你可以使用以下工具：

{tools}

要使用工具，请使用以下格式：
                                            
```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```
                                        
                                                                                        
开始！

之前的对话历史：
{chat_history}

新的输入：{input}
{agent_scratchpad}

""")
agent = create_react_agent(llm, tools, agent_prompt)

# Create a handler to call the agent
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True  # 添加这个参数以处理解析错误
)

chat_agent = RunnableWithMessageHistory(
    agent_executor,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

def generate_response(user_input):
    """
    Create a handler that calls the Conversational agent
    and returns a response to be rendered in the UI
    """
    try:
        response = chat_agent.invoke(
            {"input":user_input},
            {"configurable": {"session_id": get_session_id()}},
        )

        return response['output']
    except (CypherSyntaxError, ServiceUnavailable) as e:
        # Log the error and try the alternative tool
        st.error(f"Incident response information tool failed with error: {str(e)}")
        st.warning("Falling back to Fault Info Search tool.")

        try:
            # Manually invoke the Fault Info Search tool as a fallback
            fallback_tool = next(tool for tool in tools if tool.name == "Fault Info Search")
            return fallback_tool.func(user_input)
        except Exception as fallback_error:
            st.error(f"Fallback tool also failed: {str(fallback_error)}")
            return "抱歉，处理请求时发生了错误，所有工具均未能成功响应。"

# user_input = "你是谁？"
# response = generate_response(user_input)
# print(response)