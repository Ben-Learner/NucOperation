import streamlit as st
from llm import llm
from graph import graph

# Create the Cypher QA chain
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain

from langchain.prompts.prompt import PromptTemplate

# CYPHER_GENERATION_TEMPLATE = """
# You are an expert Neo4j Developer translating user questions into Cypher to answer questions about incident response of nuclear power plant.
# Convert the user's question based on the schema.

# Use only the provided relationship types and properties in the schema.
# Do not use any other relationship types or properties that are not provided.

# Do not return entire nodes or embedding properties.


# Schema:
# {schema}

# Question:
# {question}

# Cypher Query:
# """

CYPHER_GENERATION_TEMPLATE = """
你是一位专家级的Neo4j开发者，将用户的问题翻译成Cypher查询，以回答关于核电站事故响应的问题。
根据模式转换用户的问题。

只使用模式中提供的关系类型和属性。
不要使用任何其他未提供的关系类型或属性。
不要返回整个节点或嵌入属性。
如果查询的是规程相关的问题，可以参考以下例子：

问题：
规程中包括哪些故障

Cypher查询:
MATCH (p:Procedure{{name: '规程'}})-[:HAS_FAULT]->(f:Fault)
RETURN f.name AS Fault;

如果查询的是故障现象相关的问题，可以参考以下例子：


问题：
'1号堆蒸汽发生器传热管泄漏有哪些现象'

Cypher查询:
MATCH (f:Fault {{name: '1号堆蒸汽发生器传热管泄漏'}})-[r:HAS_PARAMETER]->(param:Parameter)
RETURN param.name AS Parameter, r.phenomenon AS Phenomenon;

模式：
{schema}

问题：
{question}


现在,请根据上述模式和问题生成Cypher查询。

Cypher查询:
"""


cypher_prompt = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)



cypher_qa = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    verbose=True,
    cypher_prompt = cypher_prompt
)