from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
import os

def build_rag_chain(file_paths):
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    for path in file_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()
        all_chunks.extend(splitter.split_documents(docs))

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(all_chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile"
    )

    prompt = PromptTemplate.from_template(
        "Use the context below to answer the question.\n\nContext: {context}\n\nQuestion: {question}"
    )

    answer_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    full_chain = RunnableParallel(
        answer=answer_chain,
        source_documents=retriever
    )

    return full_chain, retriever