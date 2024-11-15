# RAG-Based Healthcare Chatbot

An AI-powered chatbot leveraging Retrieval-Augmented Generation (RAG) to provide accurate and context-aware healthcare assistance.

## About

The RAG-Based HealthCare Chatbot is a cutting - edge AI system designed to assist users with healthcare- related queries. By Combining the power of Langchain  and LLM's  with document retrieval - based mechanism, the chatbot delivers precise and reliable information.

The Chat Bot is a created using document based retrieval using llm with vectors are stored in the PineCone db

This project focuses on:

- Helping users find answers to common health concerns.
- Providing Suggestions for first-aid and sysmpton management.
- offering resources for better understandings of health topics.

## Feature

- **RAG Architecture:** Combines retrieval-based search with generative AI to enhance response.
- **Symptom Checker:** Helps users identify potential causes of their symptoms.


## Getting Started

Follow these steps to set up the chatbot locally.


### Prerequisites

- Python 3.8+
- pip (Python package installer)
- Pinecone API
- Gemini API

### Installation

1. Clone the repository:

   ```
   git clone https://github.com/Darksoullllll/Rag-Based-Health-Care-ChatBot.git
   ```
2. Navigate to the project directory:

   ```
   cd Rag-Based-Health-Care-ChatBot
   ```
3. Create a virtual environment and activate it:

   ```
   python -m venv env
   conda activate env
   ```
4. Install the required dependencies:

   ```
   pip install -r requirements.txt

   ```

    5. Run the Chatbot


### Example Queries


- "What should I do if I have a headache?"
- "What are the symptoms of diabetes?"
- "Pain in Stomach any treatment?"

You can also customize the knowledge base to add domain-specific content for more tailored responses.



## Technology Stack

- **Python**: Core programming language.
- **RAG Framework**: Retrieval-Augmented Generation model for intelligent responses.
- **LangChain**: Manages the flow of NLP tasks.
- **Gemini/LLM**: Generates contextual and relevant answers.
- **Pinecone**: Fast and efficient retrieval of knowledge base entries.
- **Flask**: For building a web-based interface.
