# ingest.py

import os
from langchain_community.document_loaders import NotebookLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# --- Loaders and Chunkers ---
def load_and_chunk_ipynb(ipynb_path):
    loader = NotebookLoader(ipynb_path, include_outputs=True, max_output_length=500)
    documents = loader.load()
    return split_documents(documents)

def load_and_chunk_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return split_documents(documents)

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    return splitter.split_documents(documents)

# --- Main Ingestion Function ---
def ingest_documents(base_dir="./lectures"):
    all_chunks = []
    for root, _, files in os.walk(base_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            rel_path = os.path.relpath(file_path, start=base_dir)

            if filename.endswith(".ipynb"):
                print(f"Processing notebook: {rel_path}")
                chunks = load_and_chunk_ipynb(file_path)
            elif filename.endswith(".pdf"):
                print(f"Processing PDF: {rel_path}")
                chunks = load_and_chunk_pdf(file_path)
            else:
                continue

            for chunk in chunks:
                chunk.metadata["source"] = rel_path
            all_chunks.extend(chunks)

    print(f"\n✅ Total chunks created: {len(all_chunks)}")
    return all_chunks

# --- Vector Store Setup ---
def create_vector_store(chunks, persist_dir="./chroma_langchain_db"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="example_collection",
        persist_directory=persist_dir
    )
    print("Vector store saved successfully.")
    return vector_store

# --- RAG Chain Setup ---
def setup_rag_chain(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY environment variable not set. Cannot use OpenAI LLM.")
        return None

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    print("Using OpenAI GPT-3.5-turbo.")

    prompt_template = """Use the following pieces of context from my lecture notes to answer the question at the end.
If you don't know the answer based on the provided context, just say that you don't know.
Do not make up an answer or use outside knowledge.
Be concise and clear in your response.

Context:
{context}

Question: {question}

Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    return qa_chain

# --- Run Example Query ---
def run_example_query(qa_chain, query="What is confounding variables?"):
    if not qa_chain:
        print("\nRAG chain cannot be run because no LLM was successfully initialized.")
        return

    print(f"\nAsking: {query}")
    response = qa_chain.invoke({"query": query})

    print("\n--- Generated Answer ---")
    print(response["result"])

    print("\n--- Sources Used ---")
    if response["source_documents"]:
        for doc in response["source_documents"]:
            print(f"  Source: {doc.metadata.get('source', 'Unknown File')}")
    else:
        print("No source documents were retrieved for this query.")

# --- Entrypoint ---
if __name__ == "__main__":
    chunks = ingest_documents()
    vector_store = create_vector_store(chunks)
    qa_chain = setup_rag_chain(vector_store)
    run_example_query(qa_chain)
