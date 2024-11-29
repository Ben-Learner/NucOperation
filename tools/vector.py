import streamlit as st
from llm import llm, embeddings
from graph import graph

# tag::import_vector[]
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
# end::import_vector[]
# tag::import_chain[]
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
# end::import_chain[]

# tag::import_chat_prompt[]
from langchain_core.prompts import ChatPromptTemplate
# end::import_chat_prompt[]


# tag::vector[]
neo4jvector = Neo4jVector.from_existing_index(
    embeddings,                              # <1>
    graph=graph,                             # <2>
    # index_name="faultinfo",                 # <3>
    index_name="moviePlots",                 # <3>
    node_label="Information",                      # <4>
    text_node_property="name",               # <5>
    embedding_node_property="Embedding", # <6>
    retrieval_query="""
RETURN
    node.name AS text,
    score,
    {
        fault: [(node)<-[:HAS_INFORMATION]-(f:Fault) | f.name],
        information: node.name
        
    } AS metadata
"""
)

# end::vector[]

# tag::retriever[]
retriever = neo4jvector.as_retriever()
# end::retriever[]

# tag::prompt[]
instructions = (
    "You should answer the question totally from the given context."
    "If you don't know the answer, say you don't know."
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", instructions),
        ("human", "{input}"),
    ]
)
# end::prompt[]

# tag::chain[]
question_answer_chain = create_stuff_documents_chain(llm, prompt)
info_retriever = create_retrieval_chain(
    retriever, 
    question_answer_chain
)
# end::chain[]

# tag::get_movie_plot[]
def get_fault_info(input):
    return info_retriever.invoke({"input": input})
# end::get_movie_plot[]
