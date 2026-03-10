import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Path to your knowledge base
KB_FILE_PATH = "kb/knowledge_base.txt"
# Where to persist the FAISS index locally
VECTOR_STORE_PATH = "faiss_index"

# Load a free, small, local embedding model
# First run will download it (~90 MB) automatically
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def build_vector_store():
    """
    Reads the knowledge base text file, splits it into chunks,
    embeds them and saves a FAISS index to disk.
    """
    if not os.path.exists(KB_FILE_PATH):
        print(f"Knowledge base file not found at: {KB_FILE_PATH}")
        return None

    # 1. Load the text file
    loader = TextLoader(KB_FILE_PATH)
    docs = loader.load()

    # 2. Split into small overlapping chunks so retrieval is precise
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # 3. Embed and save to FAISS
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTOR_STORE_PATH)
    print(f"Vector store built with {len(chunks)} chunks and saved to '{VECTOR_STORE_PATH}'.")
    return vectorstore


def get_retriever():
    """
    Returns a FAISS retriever. Builds the index on first run.
    """
    if not os.path.exists(VECTOR_STORE_PATH):
        print("No vector store found - building for the first time...")
        vectorstore = build_vector_store()
    else:
        # Load from the saved index
        vectorstore = FAISS.load_local(
            VECTOR_STORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    if not vectorstore:
        return None
    # Return top-3 most relevant chunks
    return vectorstore.as_retriever(search_kwargs={"k": 3})


def retrieve_context(query: str) -> str:
    """
    Takes a natural-language query and returns relevant knowledge base chunks as a string.
    """
    retriever = get_retriever()
    if not retriever:
        return "Knowledge base not available."

    docs = retriever.invoke(query)
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])
    return context


if __name__ == "__main__":
    print("Testing RAG retrieval...")
    q = "What is the formula for quadratic roots?"
    result = retrieve_context(q)
    print(f"\nQuery: {q}\n\nRetrieved Context:\n{result}")
