import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate

class NeuralRAGPipeline:
    def __init__(self, api_key: str):
        self.api_key = api_key
        os.environ["OPENAI_API_KEY"] = api_key
        self.llm = ChatOpenAI(model_name="gpt-4o", temperature=0.2)
        self.embeddings = OpenAIEmbeddings()
        self.vector_db = None

    def load_index(self, folder_path="vector_store"):
        if os.path.exists(folder_path):
            self.vector_db = FAISS.load_local(
                folder_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            return True
        return False

    def query(self, user_input: str):
        if not self.vector_db:
            return "Knowledge base not initialized. Please upload documents."
        
        # Advanced Prompting for AI Engineer persona
        prompt = ChatPromptTemplate.from_template("""
        You are the Neural Insight Engine, an expert AI analytical assistant. 
        Your goal is to provide precise, data-driven answers based ONLY on the provided context.
        
        Context: {context}
        Question: {question}
        
        Rules:
        1. If the answer is not in the context, state that you don't know.
        2. Use structured formatting (bullet points, bold text) for clarity.
        3. Maintain a professional, technical, and objective tone.
        """)
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_db.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True
        )
        
        response = qa_chain.invoke({"query": user_input})
        return response["result"], response["source_documents"]
