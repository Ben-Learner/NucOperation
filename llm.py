import streamlit as st

# Create the LLM
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model=st.secrets["OPENAI_MODEL"],
    api_key=st.secrets["OPENAI_API_KEY"],
    base_url = st.secrets["BASE_URL"]
)

# messages = [
#     ("system", "You are a helpful translator. Translate the user sentence to French."),
#     ("human", "I love programming."),
# ]

# response = llm.invoke(messages)
# print(response.content)


# Create the Embedding model
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    openai_api_key = st.secrets["OPENAI_API_KEY"],
    base_url = st.secrets["BASE_URL"],
    model="text-embedding-3-small"
)

# if __name__ == "__main__":
#     messages = [
#         ("system", "You are a helpful translator. Translate the user sentence to French."),
#         ("human", "I love programming."),
#     ]
#     response = llm.invoke(messages)
#     print(response.content)
