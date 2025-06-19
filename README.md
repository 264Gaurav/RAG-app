# RAG-app

RAG (Retrieval-Augmented Generation) and basic chatbot application built using the LangChain, LangGraph, and LangSmith frameworks/tools.

## Features

- **Retrieval-Augmented Generation (RAG):** Combines retrieval techniques with generative AI to provide accurate and context-aware responses.
- **Chatbot Functionality:** Interactive chatbot powered by LangChain and LangGraph for seamless communication.
- **Modular Frameworks:** Utilizes LangSmith for advanced debugging and monitoring of AI workflows.
- **Streamed Responses:** Supports streaming responses for real-time interaction.

## Technologies Used

- **LangChain:** Framework for building applications with LLMs (Large Language Models).
- **LangGraph:** Tool for managing and executing complex workflows with AI agents.
- **LangSmith:** Framework for debugging and monitoring AI workflows.
- **Python:** Core programming language for the application.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/RAG-app.git
cd RAG-app
```

2. Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Ensure the Ollama server is running and the required models (e.g., deepseek-coder) are available.

## Usage

1. Start the chatbot application:

```bash
python chatbot.py
```

2. Interact with the chatbot by typing your queries in the terminal.

3. To exit the chatbot, type quit.

---

### Environment Variables

Ensure the following environment variables are set.

1. create .env file and insert these important key-values in that:

   OPENAI_API_KEY=sk-your-openai-key-here

   GOOGLE_API_KEY=your-gemini-api-key

# add any other secrets below

    LANGSMITH_API_KEY=your-langsmith-key

    LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"

    export LANGSMITH_TRACING=true

    export LANGSMITH_API_KEY=your-langsmith-key

# (If you also use Ollama/OpenAI, set their keys as well)

    export OLLAMA_API_KEY="your_ollama_api_key"
