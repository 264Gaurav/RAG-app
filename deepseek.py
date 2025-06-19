# chatbot.py
import os
from dotenv import load_dotenv

from langchain_ollama.llms import OllamaLLM
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Import the V2 tracing context manager
from langchain_core.tracers.context import tracing_v2_enabled

# ------------------------------------------------------------------------------
# 1. Load environment variables (OLLAMA_API_KEY, LANGSMITH_API_KEY, LANGSMITH_TRACING)
# ------------------------------------------------------------------------------

load_dotenv()  # ensures OLLAMA_API_KEY and LANGSMITH_API_KEY (and LANGSMITH_TRACING) are loaded

# ------------------------------------------------------------------------------
# 2. Initialize Ollama LLM (deepseek-r1) WITHOUT explicit tracer in code
# ------------------------------------------------------------------------------

llm = OllamaLLM(
    model="deepseek-r1",
    # No need to pass callbacks—V2 tracing will pick it up automatically if LANGSMITH_TRACING=true
)

# ------------------------------------------------------------------------------
# 3. Create an in-memory conversation buffer to store chat history
# ------------------------------------------------------------------------------

memory = ConversationBufferMemory()

# ------------------------------------------------------------------------------
# 4. Build a ConversationChain that ties together the LLM + memory
# ------------------------------------------------------------------------------

chatbot = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False,  # V2 tracing will still capture it
)

# ------------------------------------------------------------------------------
# 5. Chat loop: wrap each call in `tracing_v2_enabled`; run the chain and display the response
# ------------------------------------------------------------------------------

print("🗨️  Simple Stateful Chatbot with V2 Tracing (type 'quit' to exit)")

while True:
    user_input = input("\nYou: ").strip()
    if user_input.lower() == "quit":
        print("Chatbot: Goodbye! 👋")
        break

    # Any code inside this `with` block will be traced to LangSmith
    with tracing_v2_enabled():
        response = chatbot.predict(input=user_input)

    # Print the bot's response
    print(f"Chatbot: {response}")

























# #Chat bot using langchain and DeepSeek-R1 model via Ollama

# from langchain_core.prompts import ChatPromptTemplate
# from langchain_ollama.llms import OllamaLLM
# import streamlit as st
# import re


# st.title("Langchain-DeepSeek-R1 app")

# template = """Question: {question}

# Answer: Let's think step by step."""

# prompt = ChatPromptTemplate.from_template(template)

# #model = OllamaLLM(model="llama3.1")
# model = OllamaLLM(model="deepseek-r1")

# chain = prompt | model


# question = st.chat_input("Enter your question here")
# if question:
#     raw_output = chain.invoke({"question": question})
#     # Remove <think> tags and their content
#     cleaned_output = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()
#     st.write(raw_output)


















