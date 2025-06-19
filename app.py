import streamlit as st
import time

# Load community-contributed PDF loader
from langchain_community.document_loaders import PyPDFLoader

# For splitting long text documents into manageable chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Gemini embedding and chat LLM wrappers
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Chroma vector store for indexing embeddings
from langchain_chroma import Chroma

# RAG chain constructors
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Prompt templating for chat-style prompts
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables (e.g., GOOGLE_API_KEY) from .env
from dotenv import load_dotenv
load_dotenv()


# --- Streamlit UI setup ---
st.title("RAG Application built on Gemini Model")
# st.chat_input provides an input box for user queries
query = st.chat_input("Say something: ")


# --- Document loading & preprocessing ---
# 1. Load the PDF file into LangChain Documents
loader = PyPDFLoader("defence.pdf")
data = loader.load()  # returns a list of Document objects

# 2. Split each Document into smaller chunks (~1000 characters)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)  # list of smaller Document chunks


# --- Embedding & Vector Store Setup ---
# 3. Embed document chunks using Gemini embeddings, store in Chroma
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
)

# 4. Turn the Chroma store into a retriever that does semantic (similarity) search
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 10}  # return top 10 relevant chunks
)


# --- LLM Setup ---
# 5. Instantiate the chat-capable Gemini LLM
#    - "gemini-2.0-flash" is the chosen model
#    - temperature=0.3 controls creativity
#    - max_tokens=None/timeout=None lets the model decide length and wait time
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
    max_tokens=None,
    timeout=None
)


# --- Prompt Template for RAG ---
# 6. Define a system prompt that guides the assistant’s behavior
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

# 7. Combine system and human messages into a chat prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])


# --- RAG Execution on User Query ---
if query:
    # 8. Build a simple "stuff" chain that stuffs all retrieved docs into the prompt
    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    # 9. Create the retrieval-augmented chain: retriever + QA chain
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # 10. Run the chain: embed the query, fetch docs, then generate an answer
    response = rag_chain.invoke({"input": query})

    # 11. Display the answer in the Streamlit app
    st.write(response["answer"])








































# import streamlit as st
# import time
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_chroma import Chroma
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate




# from dotenv import load_dotenv
# load_dotenv()


# st.title("RAG Application built on Gemini Model")

# loader = PyPDFLoader("defence.pdf")
# data = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
# docs = text_splitter.split_documents(data)


# vectorstore = Chroma.from_documents(documents=docs, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

# retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",temperature=0.3,max_tokens=None,timeout=None)


# query = st.chat_input("Say something: ")
# prompt = query

# system_prompt = (
#     "You are an assistant for question-answering tasks. "
#     "Use the following pieces of retrieved context to answer "
#     "the question. If you don't know the answer, say that you "
#     "don't know. Use three sentences maximum and keep the "
#     "answer concise."
#     "\n\n"
#     "{context}"
# )

# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         ("human", "{input}"),
#     ]
# )

# if query:
#     question_answer_chain = create_stuff_documents_chain(llm, prompt)
#     rag_chain = create_retrieval_chain(retriever, question_answer_chain)

#     response = rag_chain.invoke({"input": query})
#     #print(response["answer"])

#     st.write(response["answer"])
