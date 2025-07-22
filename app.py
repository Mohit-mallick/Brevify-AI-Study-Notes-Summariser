import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
import google.generativeai as genai
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import re
from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp
from urllib.parse import urlparse
import time

# Load environment variables
load_dotenv()

# Configure API keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "your_api_key_here")
genai.configure(api_key=GOOGLE_API_KEY)

# Page configuration
st.set_page_config(
    page_title="モヒット",
    page_icon="🐢",
    layout="wide"
)

# Custom CSS for clean design
st.markdown("""
<style>
    .main {
        padding: 1rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        color: white;
    }
    
    .header h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    
    .header p {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    .content-section {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    
    .result-container {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin-top: 1rem;
    }
    
    .info-box {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    
    .error-box {
        background: #ffebee;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #f44336;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    }
    
    .sidebar .stSelectbox {
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_content' not in st.session_state:
    st.session_state.processed_content = False
if 'content_type' not in st.session_state:
    st.session_state.content_type = None

# Header
st.markdown("""
<div class="header">
    <h1>🗻🌸 Brevify – Study Notes Summariser 🌸🗻</h1>
    <p>Summarize PDFs, web pages, and study material with — with Brevify.</p>
</div>
""", unsafe_allow_html=True)

# Helper Functions
def extract_pdf_text(pdf_files):
    """Extract text from PDF files"""
    text = ""
    for pdf in pdf_files:
        try:
            reader = PdfReader(pdf)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {str(e)}")
    return text

def create_text_chunks(text, chunk_size=10000, overlap=1000):
    """Split text into chunks for processing"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
    )
    return splitter.split_text(text)

@st.cache_resource
def build_vector_store(text_chunks):
    """Create vector store from text chunks"""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        return True
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return False

def setup_qa_chain():
    """Initialize the QA chain"""
    template = """
    Answer the question based on the provided context. If the information is not available 
    in the context, please say "I don't have enough information to answer that question."
    
    Context: {context}
    Question: {question}
    
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=GOOGLE_API_KEY
    )
    
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)

def get_answer(question):
    """Get answer from processed documents"""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        
        vector_store = FAISS.load_local(
            "faiss_index", 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        docs = vector_store.similarity_search(question, k=4)
        chain = setup_qa_chain()
        
        response = chain(
            {"input_documents": docs, "question": question},
            return_only_outputs=True
        )
        
        return response["output_text"]
        
    except Exception as e:
        return f"Error processing question: {str(e)}"

def extract_youtube_id(url):
    """Extract YouTube video ID from URL"""
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_youtube_transcript(video_id):
    """Get YouTube video transcript"""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Try English first, then any available language
        try:
            transcript = transcript_list.find_transcript(['en'])
        except:
            transcript = next(iter(transcript_list))
        
        transcript_data = transcript.fetch()
        return " ".join([item['text'] for item in transcript_data])
        
    except Exception:
        return None

def get_youtube_metadata(video_id):
    """Get YouTube video information"""
    try:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(
                f"https://www.youtube.com/watch?v={video_id}", 
                download=False
            )
            
            return {
                'title': info.get('title', 'Unknown'),
                'description': info.get('description', ''),
                'duration': info.get('duration', 0),
                'view_count': info.get('view_count', 0),
                'uploader': info.get('uploader', 'Unknown'),
                'upload_date': info.get('upload_date', 'Unknown')
            }
            
    except Exception:
        return None

def extract_web_content(url):
    """Extract content from web pages"""
    try:
        # Set headers to mimic a real browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        # Make request with timeout
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
            script.decompose()
        
        # Extract text content
        text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Limit content length to avoid overwhelming the model
        if len(text) > 20000:
            text = text[:20000] + "..."
        
        return text
        
    except requests.RequestException as e:
        st.error(f"Error fetching web page: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error parsing web page: {str(e)}")
        return None

def analyze_with_gemini(content, question=None):
    """Analyze content using Gemini"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        if question:
            prompt = f"""
            Please answer the following question based on the provided content:
            
            Question: {question}
            
            Content: {content[:8000]}
            
            Provide a clear, detailed answer based on the content.
            """
        else:
            prompt = f"""
            Please provide a comprehensive summary of the following content:
            
            {content[:8000]}
            
            Include:
            - Main topics and themes
            - Key points and insights
            - Important details
            """
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"Error analyzing content: {str(e)}"

def format_duration(seconds):
    """Format duration in human readable format"""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes}m {secs}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m"

# Sidebar
with st.sidebar:
    st.markdown("## 🎯 Select Content Type")
    
    content_type = st.selectbox(
        "What would you like to analyze?",
        ["📄 PDF Documents", "🔗 YouTube Videos", "🌐 Web Pages"],
        key="content_type_selector"
    )
    
    st.markdown("---")
    
    # PDF Section
    if content_type == "📄 PDF Documents":
        st.markdown("### 📄 Upload PDFs")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            accept_multiple_files=True,
            type=['pdf'],
            key="pdf_uploader"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
            
            if st.button("🚀 Process Documents", key="process_pdf"):
                with st.spinner("Processing documents..."):
                    # Extract text
                    text = extract_pdf_text(uploaded_files)
                    
                    if text.strip():
                        # Create chunks
                        chunks = create_text_chunks(text)
                        
                        # Build vector store
                        if build_vector_store(chunks):
                            st.session_state.processed_content = True
                            st.session_state.content_type = "pdf"
                            st.success("✅ Documents processed successfully!")
                        else:
                            st.error("❌ Failed to process documents")
                    else:
                        st.error("❌ No text found in the uploaded files")
    
    # YouTube Section
    elif content_type == "🔗 YouTube Videos":
        st.markdown("### 🔗 YouTube URL")
        youtube_url = st.text_input(
            "Enter YouTube URL:",
            placeholder="https://www.youtube.com/watch?v=...",
            key="youtube_input"
        )
        
        if youtube_url:
            video_id = extract_youtube_id(youtube_url)
            if video_id:
                st.success("✅ Valid YouTube URL")
                st.session_state.youtube_id = video_id
                st.session_state.content_type = "youtube"
                
                # Show thumbnail
                st.image(
                    f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
                    use_container_width=True
                )
            else:
                st.error("❌ Invalid YouTube URL")
    
    # Web Page Section
    elif content_type == "🌐 Web Pages":
        st.markdown("### 🌐 Web Page URL")
        web_url = st.text_input(
            "Enter web page URL:",
            placeholder="https://example.com",
            key="web_input"
        )
        
        if web_url:
            if web_url.startswith(('http://', 'https://')):
                st.success("✅ Valid URL")
                st.session_state.web_url = web_url
                st.session_state.content_type = "web"
                
                # Show domain info
                domain = urlparse(web_url).netloc
                st.info(f"🌐 Domain: {domain}")
            else:
                st.error("❌ Please include http:// or https://")
    
    st.markdown("---")
    st.markdown("### 📈 **Supported Formats**")
    st.markdown("""
    - **Documents**: PDF files
    - **URLs**: YouTube, Web pages
    - **Languages**: Multi-language support
    """)

# Main Content Area
st.markdown('<div class="content-section">', unsafe_allow_html=True)

# PDF Analysis
if content_type == "📄 PDF Documents":
    st.markdown("## 📄 PDF Document Analysis")
    
    if st.session_state.get('processed_content') and st.session_state.get('content_type') == 'pdf':
        st.markdown('<div class="success-box">✅ Documents are ready for analysis!</div>', unsafe_allow_html=True)
        
        # Question input
        question = st.text_input(
            "Ask a question about your documents:",
            placeholder="What are the main topics discussed in the documents?",
            key="pdf_question"
        )
        
        if question:
            if st.button("🔍 Get Answer", key="pdf_analyze"):
                with st.spinner("Analyzing documents..."):
                    answer = get_answer(question)
                    
                    st.markdown(f"""
                    <div class="result-container">
                        <h4>🤖 Answer:</h4>
                        <p>{answer}</p>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box">👆 Upload PDF files in the sidebar to get started</div>', unsafe_allow_html=True)

# YouTube Analysis
elif content_type == "🔗 YouTube Videos":
    st.markdown("## 🔗 YouTube Video Analysis")
    
    if st.session_state.get('youtube_id'):
        video_id = st.session_state.youtube_id
        
        # Question input
        question = st.text_area(
            "What would you like to know about this video?",
            placeholder="Leave empty for a general summary, or ask specific questions...",
            height=100,
            key="youtube_question"
        )
        
        if st.button("🔍 Analyze Video", key="youtube_analyze"):
            with st.spinner("Analyzing video..."):
                # Get video metadata
                metadata = get_youtube_metadata(video_id)
                
                if metadata:
                    # Try to get transcript
                    transcript = get_youtube_transcript(video_id)
                    
                    if transcript:
                        content = f"Title: {metadata['title']}\n\nTranscript:\n{transcript}"
                        analysis = analyze_with_gemini(content, question)
                    else:
                        content = f"Title: {metadata['title']}\n\nDescription:\n{metadata['description']}"
                        analysis = analyze_with_gemini(content, question)
                        st.warning("⚠️ Transcript not available. Analysis based on title and description.")
                    
                    # Display results
                    st.markdown(f"""
                    <div class="result-container">
                        <h4>🎥 Analysis Results:</h4>
                        <p>{analysis}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Video statistics
                    st.markdown("### 📊 Video Statistics")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>👁️ Views</h3>
                            <p>{metadata['view_count']:,}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>⏱️ Duration</h3>
                            <p>{format_duration(metadata['duration'])}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>📺 Channel</h3>
                            <p>{metadata['uploader']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.error("❌ Unable to fetch video information")
    else:
        st.markdown('<div class="info-box">👆 Enter a YouTube URL in the sidebar to get started</div>', unsafe_allow_html=True)

# Web Page Analysis
elif content_type == "🌐 Web Pages":
    st.markdown("## 🌐 Web Page Analysis")
    
    if st.session_state.get('web_url'):
        web_url = st.session_state.web_url
        
        # Question input
        question = st.text_area(
            "What would you like to know about this web page?",
            placeholder="Leave empty for a general summary, or ask specific questions...",
            height=100,
            key="web_question"
        )
        
        if st.button("🔍 Analyze Web Page", key="web_analyze"):
            with st.spinner("Analyzing web page..."):
                # Extract content
                content = extract_web_content(web_url)
                
                if content:
                    # Analyze content
                    analysis = analyze_with_gemini(content, question)
                    
                    # Display results
                    st.markdown(f"""
                    <div class="result-container">
                        <h4>🌐 Analysis Results:</h4>
                        <p>{analysis}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Page information
                    st.markdown("### 📊 Page Information")
                    domain = urlparse(web_url).netloc
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>🌐 Domain</h3>
                            <p>{domain}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>📄 Content Length</h3>
                            <p>{len(content):,} chars</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="error-box">
                        <h4>❌ Unable to extract content</h4>
                        <p>This could happen due to:</p>
                        <ul>
                            <li>Website blocks automated requests</li>
                            <li>Content requires JavaScript to load</li>
                            <li>Website has anti-scraping protection</li>
                            <li>Network connectivity issues</li>
                        </ul>
                        <p><strong>Try:</strong> Copy and paste the text content directly, or use a different URL.</p>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box">👆 Enter a web page URL in the sidebar to get started</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🤖 <strong> Brevify – Study Notes Summariser </strong> | Made by モヒット</p>
</div>
""", unsafe_allow_html=True)