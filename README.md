# DocuMind-RAG

# DocuMind: Enterprise Financial RAG Analyst

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-v0.2-green)
![Groq](https://img.shields.io/badge/Inference-Groq%20LPU-orange)
![ChromaDB](https://img.shields.io/badge/Vector%20DB-Chroma-purple)

**DocuMind** is an autonomous AI agent designed to analyze dense financial documents (like SEC 10-K filings) and answer complex investor queries with high precision. 

Unlike standard chatbots, DocuMind uses **Retrieval Augmented Generation (RAG)** to ground its answers in factual data, providing specific **page-level citations** for every claim it makes.

---

## Key Features

* **Sub-Second Latency:** Powered by **Groq's LPU (Language Processing Unit)**, delivering Llama-3 inference at >300 tokens/second.
* **Long-Context Understanding:** Uses **Recursive Character Chunking** to preserve the semantic structure of financial tables and multi-paragraph risk factors.
* **Auditability (Source Citations):** Every answer includes a "Sources" section, tracing the AI's reasoning back to the specific PDF page number.
* **Private & Free Embeddings:** Uses local HuggingFace embeddings (`all-MiniLM-L6-v2`) running on-device, ensuring document vectors are not sent to third-party APIs.
* **Persistent Memory:** Automatically caches vectors in `ChromaDB` to prevent re-indexing the same document twice.

---

## Tech Stack

| Component | Technology | Reasoning |
| :--- | :--- | :--- |
| **Orchestration** | [LangChain](https://www.langchain.com/) | For building the RAG retrieval pipeline and context window management. |
| **LLM Inference** | [Groq](https://groq.com/) (Llama-3-8b) | Chosen for ultra-low latency, crucial for real-time analyst tools. |
| **Vector DB** | [ChromaDB](https://www.trychroma.com/) | Lightweight, open-source vector store for semantic search. |
| **Embeddings** | [HuggingFace](https://huggingface.co/) | `all-MiniLM-L6-v2` for state-of-the-art sentence similarity without API costs. |
| **Data Ingestion** | `PyPDF` | For extracting raw text from unstructured PDF layouts. |

---

## Installation

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/yourusername/DocuMind.git](https://github.com/yourusername/DocuMind.git)
    cd DocuMind
    ```

2.  **Install Dependencies**
    ```bash
    pip install langchain langchain-groq langchain-huggingface langchain-chroma pypdf sentence-transformers
    ```

3.  **Set up API Keys**
    * Get a free API Key from [Groq Console](https://console.groq.com/).
    * Set it in your environment (or use Colab User Secrets):
    ```bash
    export GROQ_API_KEY="gsk_..."
    ```

---

## Usage

1.  **Prepare Data:** Place your target PDF (e.g., `apple_10k.pdf`) in the project root.
2.  **Run the Agent:**
    ```bash
    python documind.py
    ```
3.  **Interact:**
    ```text
    Initializing DocuMind Financial Analyst...
    Loading apple_10k.pdf...
    Knowledge Base Ready!

    [Investor]: What are the primary foreign exchange risks mentioned?
    
    Analyzing...
    
    Answer (0.8s):
    The company faces risks related to fluctuations in the value of the U.S. dollar 
    relative to foreign currencies, particularly the Euro, Chinese Renminbi, and Japanese Yen. 
    This impacts net sales and operating margins...

    Sources:
    - Page 42: ...foreign currency exchange rates...
    - Page 18: ...international operations risks...
    ```

---

## Architecture

```mermaid
graph LR
    A[PDF Document] -->|PyPDF Loader| B[Raw Text]
    B -->|Recursive Splitter| C[Chunks]
    C -->|HuggingFace Embeddings| D[(Chroma Vector DB)]
    
    U[User Query] -->|Embed| E[Query Vector]
    E -->|Similarity Search| D
    D -->|Top-k Context| F[Llama-3 Prompt]
    U --> F
    F -->|Groq Inference| G[Final Answer + Citations]
