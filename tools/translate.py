from llm import llm
from langchain.prompts.prompt import PromptTemplate

TRANSLATE_TEMPLATE = """
你是一位专家级的Neo4j开发者，按照下述示例将用户的问题翻译成Cypher查询需要的关键词，以回答关于核电站事故响应的问题。
根据模式转换用户的问题。

只使用模式中提供的关系类型和属性。
不要使用任何其他未提供的关系类型或属性。

不要返回整个节点或嵌入属性。获取现象时的Cyber语句应该从故障与参数的边上获取，
如：MATCH (f:Fault)-[r:HAS_PARAMETER]->(param:Parameter)
RETURN param.name AS Parameter, r.phenomenon AS Phenomenon;注意：[r:HAS_PARAMETER]是不可变的。

模式：
{schema}

问题：
{question}


现在,请根据上述模式和问题生成Cypher查询。

Cypher查询:






"""

def transCyber():
    """
    提取用户问题关键词，用于编写Cyber语言
    """
