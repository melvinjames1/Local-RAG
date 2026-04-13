import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA


def load_documents(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)


def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    vectorstore.persist()
    return vectorstore


def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )


def create_qa_chain(vectorstore):
    llm = Ollama(
        model="mistral",
        temperature=0
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True
    )

    return qa_chain


def main():
    pdf_path = "data/your_pdf.pdf"

    print("Loading PDF...")
    docs = load_documents(pdf_path)

    print("Chunking...")
    chunks = split_documents(docs)

    if not os.path.exists("./chroma_db"):
        print("Creating vector DB...")
        vectorstore = create_vector_store(chunks)
    else:
        print("Loading existing DB...")
        vectorstore = load_vector_store()

    print("Starting local LLM...")
    qa_chain = create_qa_chain(vectorstore)

    print("\nRAG ready (LOCAL). Type 'exit' to quit.\n")

    while True:
        query = input("Question: ")

        if query.lower() == "exit":
            break

        result = qa_chain.invoke({"query": query})

        print("\nAnswer:\n", result["result"])

        print("\nSources:")
        for doc in result["source_documents"]:
            print("-", doc.metadata.get("source"))

        print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    main()
