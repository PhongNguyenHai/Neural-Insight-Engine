import streamlit as st
import os
from core.rag_pipeline import NeuralRAGPipeline
from utils.document_processor import DocumentProcessor
from dotenv import load_dotenv

# App Config
load_dotenv()
st.set_page_config(page_title="Neural Insight Engine", page_icon="🧠", layout="wide")

# Styling
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #4CAF50; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("🧠 Neural Insight Engine")
st.subheader("Advanced RAG System for Document Intelligence")

# Sidebar
with st.sidebar:
    st.header("🔐 Authentication")
    api_key = st.text_input("Enter OpenAI API Key", type="password")
    
    st.divider()
    st.header("📁 Data Ingestion")
    uploaded_files = st.file_uploader("Upload Documents (PDF/TXT)", accept_multiple_files=True)
    
    if st.button("🚀 Process \u0026 Index"):
        if not api_key:
            st.error("API Key required!")
        elif uploaded_files:
            if not os.path.exists("temp_data"): os.makedirs("temp_data")
            for f in uploaded_files:
                with open(os.path.join("temp_data", f.name), "wb") as buffer:
                    buffer.write(f.getvalue())
            
            processor = DocumentProcessor(api_key)
            num_chunks = processor.process_folder("temp_data")
            st.success(f"Indexed {num_chunks} chunks successfully!")
        else:
            st.warning("No files uploaded.")

# Main Interface
if api_key:
    pipeline = NeuralRAGPipeline(api_key)
    has_index = pipeline.load_index()
    
    if has_index:
        query = st.text_input("🔍 Ask a complex question about your documents:")
        
        if query:
            with st.spinner("Thinking..."):
                answer, sources = pipeline.query(query)
                
                st.markdown("### 💡 Intelligence Report")
                st.write(answer)
                
                with st.expander("📚 View Evidence (Sources)"):
                    for doc in sources:
                        st.info(f"**Source:** {doc.metadata.get('source', 'Unknown')}\n\n**Content Snippet:** {doc.page_content[:300]}...")
    else:
        st.info("👈 Please index your documents first via the sidebar.")
else:
    st.warning("👈 Please enter your OpenAI API Key to start.")
