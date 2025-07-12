# 📚 Lecture RAG Assistant

This project is a Retrieval-Augmented Generation (RAG) pipeline for querying lecture notes stored in `.ipynb` (Jupyter Notebook) and `.pdf` format using a conversational interface powered by LLMs. The system uses LangChain, ChromaDB, and OpenAI (or other models) to deliver accurate, source-grounded answers from your personal lecture materials.

---

## 📁 Project Structure

```
lecture_rag/
├── app.py                  # Streamlit dashboard for Q&A
├── ingest.py               # Extracts & chunks lecture files, saves to ChromaDB
├── qa_pipeline.py          # Loads retriever + LLM to create QA chain
├── lectures/               # Directory with lecture content (PDFs, Notebooks)
├── chroma_langchain_db/    # Persisted Chroma vector DB (auto-created)
├── README.md               # Project documentation
```

---

## ⚙️ Setup Instructions

### 1. 🔧 Install Dependencies

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt` yet, here's a quick list:

```bash
pip install langchain openai chromadb streamlit sentence-transformers PyMuPDF
```

---

### 2. 👅 Ingest Lecture Notes

Make sure your `./lectures/` directory contains `.ipynb` and/or `.pdf` files (subfolders supported).

Then run:

```bash
python ingest.py
```

This will:

* Recursively scan the lecture folder
* Chunk the content intelligently
* Store the chunks in a Chroma vector DB using HuggingFace embeddings

---

### 3. 🤖 Launch the Q\&A Dashboard

```bash
streamlit run app.py
```

This will start a local web app where you can:

* Ask questions about your lecture content
* See answers grounded in real chunks
* View the exact source of each response

---

## 🔑 API Key

To use OpenAI models (like `gpt-3.5-turbo`), set your API key in your environment:

```bash
export OPENAI_API_KEY="your_openai_key_here"
```

If no API key is found, the app will notify you and skip LLM invocation.

---

## 🧐 How It Works

1. **Ingestion** (`ingest.py`)

   * Loads `.ipynb` with `NotebookLoader` and `.pdf` with `PyPDFLoader`
   * Splits documents using `RecursiveCharacterTextSplitter`
   * Embeds chunks with `all-mpnet-base-v2`
   * Saves everything to a ChromaDB collection

2. **Retrieval** (`qa_pipeline.py`)

   * Loads persisted vector store
   * Wraps it with a retriever and LLM using LangChain's `RetrievalQA`

3. **Interaction** (`app.py`)

   * Accepts user questions
   * Retrieves relevant lecture chunks
   * Sends them + question to the LLM with a helpful prompt
   * Displays result and source document metadata

---

## 📌 TODO (Optional Features)

* [ ] Add support for `.md` or `.html` files
* [ ] Use local LLM (e.g. Mistral, Llama.cpp) via HuggingFace pipeline
* [ ] Add upload feature in Streamlit for ad-hoc notes
* [ ] Rerank retrieved chunks for better answers

---

## 📣 Credits

* Built using [LangChain](https://www.langchain.com/)
* Embeddings from [SentenceTransformers](https://www.sbert.net/)
* PDF parsing by [PyMuPDF](https://pymupdf.readthedocs.io/)
* UI via [Streamlit](https://streamlit.io/)

---

## 💬 Example Questions

* "What is a confounding variable?"
* "How does backpropagation work?"
* "What are the key assumptions of linear regression?"
