# Moroccan Constitution Q\&A (RAG System)

## 📖 Overview

This project implements an **interactive Question-Answering system** dedicated to the **Moroccan Constitution of 2011**.
It leverages **Retrieval-Augmented Generation (RAG)** to combine:

* The **factual accuracy** of retrieval from the official constitution text, and
* The **fluency of Large Language Models (LLMs)** for natural responses.

Built with:

* **Python**
* **Sentence-Transformers** (all-MiniLM-L6-v2)
* **FAISS** (vector search)
* **Hugging Face Transformers** (Flan-T5-small)
* **Streamlit** (for the user interface)

---

## ⚙️ Features

* Natural language Q\&A on the Moroccan Constitution.
* Retrieval of relevant constitutional articles with **exact citations**.
* **Semantic search** powered by embeddings + FAISS.
* **Interactive web app** built with Streamlit.
* Conversation history with sources for traceability.

---

## 📂 Project Structure

```
├── app.py                 # Streamlit interface
├── retriever.py           # Retrieval + FAISS logic
├── generator.py           # LLM integration (Flan-T5-small)
├── preprocessing.py       # PDF ingestion, cleaning, chunking
├── data/                  # Contains Moroccan Constitution (PDF)
├── requirements.txt       # Dependencies
└── README.md              # Documentation
```

---

## 🚀 Installation & Usage

1. Clone the repo:

   ```bash
   git clone https://github.com/jawad-elkharrati/MoroccanConstitution-QA-RAG.git
   cd MoroccanConstitution-QA-RAG
   ```

2. Create a virtual environment & install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:

   ```bash
   streamlit run app.py
   ```

4. Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📊 Future Improvements

* Add **multilingual support** (Arabic & French).
* Extend to other legal documents (laws, decrees).
* Optimize retrieval with advanced FAISS indexing.

---

## 👨‍💻 Authors

* Jawad El Kharrati
* Youssef Guini
* Shalom Junior Nukunu
  *(ENSA Oujda, 2024–2025)*

Supervisor: Prof. Abdelmounaim Kerkri

---

## 📜 License

MIT License
