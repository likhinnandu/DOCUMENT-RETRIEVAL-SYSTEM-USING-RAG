# DOCUMENT RETRIEVAL SYSTEM USING RAG

## Problem Statement
Organizations and individuals often deal with large volumes of documents, making it difficult to extract specific, context-relevant information quickly. Traditional keyword-based search systems often fail to understand the semantic meaning of queries, leading to inaccurate or irrelevant results. This project addresses the need for an intelligent question-answering system that can intuitively understand natural language queries and retrieve precise, context-aware answers from a custom collection of PDF documents, significantly reducing the time spent searching for information.

## Architecture
The system employs a **Retrieval-Augmented Generation (RAG)** architecture to deliver accurate responses:
1. **Document Ingestion**: PDF documents placed in the `Artifacts` directory are loaded using `PyPDFDirectoryLoader`.
2. **Text Splitting**: The loaded documents are divided into smaller, manageable chunks using LangChain's `RecursiveCharacterTextSplitter`.
3. **Embedding Generation**: These text chunks are converted into vector embeddings using Google Generative AI Embeddings (`models/embedding-001`).
4. **Vector Store**: The embeddings are subsequently stored and indexed in a **FAISS** vector database, allowing for fast and efficient similarity search.
5. **Retrieval & Answer Generation**: When a user inputs a question, the application queries the FAISS vector store to retrieve the most relevant document chunks. These chunks are provided as context to a Large Language Model (Groq `Llama3-8b-8192`), which synthesizes a precise answer based exclusively on the provided context.

## Project Structure
```text
DOCUMENT RETRIEVAL SYSTEM USING RAG/
├── Artifacts/              # Directory to place input PDF documents for ingestion
├── .env                    # Environment variables file (API keys)
├── app.py                  # Main Streamlit application file containing the logic
├── requirements.txt        # Required Python libraries and dependencies
├── README.md               # Project documentation
└── LICENSE                 # Project license
```

## Setup and Installation

Follow these steps to configure the project locally:

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd "DOCUMENT RETRIEVAL SYSTEM USING RAG"
   ```

2. **Create a Virtual Environment** (Recommended)
   ```bash
   python -m venv venv
   
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **API Key Setup**
   The application requires API keys from Groq and Google.
   - Get your Groq API key: [Groq Console](https://console.groq.com/keys)
   - Get your Google API key: [Google AI Studio](https://aistudio.google.com/app/apikey)
   
   Create a `.env` file in the root directory and add your keys:
   ```dotenv
   API_KEY=your_groq_api_key_here
   GOOGLE_API_KEY=your_google_api_key_here
   ```

5. **Prepare Documents**
   Create a folder named `Artifacts` in the root directory (if it doesn't exist) and place all the PDF documents you want the system to process inside this folder.

## Commands to Run the Program

Once the setup is complete and your PDFs are in the `Artifacts` folder, you can start the application using Streamlit:

```bash
streamlit run app.py
```

After running the command, open your web browser to the provided local URL (usually `http://localhost:8501`). On the web interface, click **"Ingest the Data into Vector Store"** first to build the embeddings, and then you can start asking questions from your documents!
