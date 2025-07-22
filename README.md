# 📘 Brevify – AI Study Notes Summariser

**Brevify** is an AI-powered study assistant that summarizes **PDFs**, **YouTube videos**, and **web articles** using **Retrieval-Augmented Generation (RAG)**.

## 🖼️ Screenshot
![App Screenshot](https://i.ibb.co/Wv8yxtGw/Screenshot-2025-07-22-212157.png)



## 🚀 Features

- 📄 PDF Summarization  
- 📺 YouTube Video Transcript Summarization  
- 🌐 Web Page Content Summarization  
- 🔍 RAG Pipeline using **LangChain**, **FAISS**, and **Gemini API**

---

## 🛠 Tech Stack

- Python, Streamlit  
- LangChain + FAISS (Vector DB)  
- Gemini API (Google Generative AI)  
- PyPDF2, YouTubeTranscriptApi, BeautifulSoup  

---

## 🧩 Installation

```bash
git clone <your-repo-url>
cd brevify
pip install -r requirements.txt

⚙️ Setup
Create a .env file in the root directory
Add your API key:
GOOGLE_API_KEY=your_google_api_key
