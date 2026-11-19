
# AI Knowledge Assistant 🤖

An AI-powered PDF chatbot built using Retrieval-Augmented Generation (RAG), Google Gemini, ChromaDB, and Streamlit.

## Overview

AI Knowledge Assistant allows users to upload PDF documents and ask questions about their content using natural language.

The application retrieves relevant information from the uploaded document and uses Google Gemini to generate context-aware answers.

## Features

- Upload and process PDF documents
- Extract text and split it into chunks
- Generate semantic embeddings
- Store and retrieve embeddings using ChromaDB
- Generate AI-powered answers using Google Gemini
- Display retrieved PDF page references
- ChatGPT-style conversational interface
- Maintain chat history during the session
- Support follow-up questions using conversation context

## Tech Stack

- Python
- Streamlit
- Google Gemini API
- ChromaDB
- Sentence Transformers
- LangChain Text Splitters
- PyMuPDF

## How It Works

1. User uploads a PDF document.
2. PyMuPDF extracts text from each page.
3. The document is divided into overlapping chunks.
4. Sentence Transformers generates vector embeddings.
5. ChromaDB stores and searches the embeddings.
6. Relevant document chunks are sent to Google Gemini.
7. Gemini generates an answer using the retrieved context.
8. The application displays the answer and retrieved page references.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/ai-knowledge-assistant.git
cd ai-knowledge-assistant
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your_api_key_here"
```

Run the application:

```bash
streamlit run app.py
```

## Limitations

- Scanned PDFs without selectable text may require OCR.
- Answers depend on the quality of retrieved document chunks.
- Chat history is maintained only during the active session.
- Follow-up question retrieval can be improved further.

## Future Improvements

- Multiple PDF support
- Improved conversational retrieval
- Persistent chat history
- Cloud deployment
- Advanced source citations

## Author

Built as an AI/ML portfolio project.
