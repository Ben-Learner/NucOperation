import streamlit as st
from utils import write_message
from llm import llm 
from agent import generate_response

# Page Config
st.set_page_config("Ebert", page_icon=":movie_camera:")

# Set up Session State
# if "messages" not in st.session_state:
#     st.session_state.messages = [
#         {"role": "assistant", "content": "Hi, I'm the NucOperation Chatbot!  How can I help you?"},
#     ]

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "你好，我是 NucOperation 聊天机器人！ 有什么可以帮您？"},
    ]

# Submit handler
def handle_submit(message):
    """
    Submit handler:

    You will modify this method to talk with an LLM and provide
    context using data from Neo4j.
    """

    # Handle the response
    with st.spinner('Thinking...'):
        # # TODO: Replace this with a call to your LLM
        from time import sleep
        sleep(1)
        response = generate_response(message)
        print(f"response:{response}")
        write_message('assistant', response)


# Display messages in Session State
for message in st.session_state.messages:
    write_message(message['role'], message['content'], save=False)

# Handle any user input
if question := st.chat_input("What is up?"):
    # Display user message in chat message container
    write_message('user', question)

    # Generate a response
    handle_submit(question)
