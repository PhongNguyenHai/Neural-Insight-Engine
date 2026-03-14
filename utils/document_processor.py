import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

class DocumentProcessor:
    def __init__(self, api_key: str):
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len,
            add_start_index=True,
        )

    def process_folder(self, data_path: str):
        documents = []
        for file in os.listdir(data_path):
            file_path = os.path.join(data_path, file)
            if file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif file.endswith(".txt"):
                loader = TextLoader(file_path)
                documents.extend(loader.load())
        
        chunks = self.text_splitter.split_documents(documents)
        vector_db = FAISS.from_documents(chunks, self.embeddings)
        vector_db.save_local("vector_store")
        return len(chunks)
