import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
KB_FILE_PATH = 'kb/knowledge_base.txt'
VECTOR_STORE_PATH = 'faiss_index'
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')


def build_vector_store():
    if not os.path.exists(KB_FILE_PATH):
        print(f'Knowledge base file not found at: {KB_FILE_PATH}')
        return None
    loader = TextLoader(KB_FILE_PATH)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTOR_STORE_PATH)
    print(
        f"Vector store built with {len(chunks)} chunks and saved to '{VECTOR_STORE_PATH}'."
        )
    return vectorstore


def get_retriever():
    if not os.path.exists(VECTOR_STORE_PATH):
        print('No vector store found - building for the first time...')
        vectorstore = build_vector_store()
    else:
        vectorstore = FAISS.load_local(VECTOR_STORE_PATH, embeddings,
            allow_dangerous_deserialization=True)
    if not vectorstore:
        return None
    return vectorstore.as_retriever(search_kwargs={'k': 3})


def retrieve_context(query: str) ->str:
    retriever = get_retriever()
    if not retriever:
        return 'Knowledge base not available.'
    docs = retriever.invoke(query)
    context = '\n\n---\n\n'.join([doc.page_content for doc in docs])
    return context


if __name__ == '__main__':
    print('Testing RAG retrieval...')
    q = 'What is the formula for quadratic roots?'
    result = retrieve_context(q)
    print(f'\nQuery: {q}\n\nRetrieved Context:\n{result}')
