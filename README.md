# 🧠 Neural Insight Engine

An advanced Retrieval-Augmented Generation (RAG) system engineered for high-precision document intelligence. This engine implements hybrid search strategies, agentic re-ranking, and modular AI architecture to provide deep insights from complex datasets.

---

### ✨ Key Features
- **Hybrid Search**: Combines Dense Vector Search (Semantic) with Sparse Keyword Search (BM25) for maximum retrieval accuracy.
- **Agentic Re-ranking**: Uses LLM-based re-rankers to evaluate the relevance of retrieved context before generation.
- **Multi-Format Intelligence**: Native support for PDF, DOCX, and Markdown with intelligent structural parsing.
- **Vector Persistence**: Integrated with FAISS/ChromaDB for fast, persistent semantic indexing.
- **Interactive Dashboard**: Professional Streamlit UI for real-time querying and source visualization.

---

### 🚀 Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/PhongNguyenHai/Neural-Insight-Engine.git
   cd Neural-Insight-Engine
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment**
   Create a `.env` file:
   ```env
   OPENAI_API_KEY=your_key_here
   PINECONE_API_KEY=optional_for_cloud
   ```

4. **Launch the Engine**
   ```bash
   streamlit run app.py
   ```

---

### 🛠️ Technical Stack
- **Orchestration**: LangChain / LlamaIndex
- **LLMs**: OpenAI GPT-4o / Gemini 1.5 Pro
- **Vector DB**: FAISS (Local) / Pinecone (Cloud)
- **Embeddings**: HuggingFace Transformers / OpenAI
- **UI**: Streamlit

---

### 📜 License
MIT
