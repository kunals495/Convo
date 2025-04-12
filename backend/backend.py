import os
import shutil
import certifi
import requests
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import UnstructuredURLLoader

# 🔹 Hardcoded Google API Key (⚠️ NOT SECURE for production)
GOOGLE_API_KEY = ""

# Set API key for Google Gemini
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Force requests to use the certifi certificate bundle
requests.packages.urllib3.util.ssl_.DEFAULT_CA_BUNDLE_PATH = certifi.where()
os.environ['SSL_CERT_FILE'] = certifi.where()

# 🔹 Initialize FastAPI app
app = FastAPI()

# 🔹 Enable CORS (Allow Frontend Requests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Only allow React frontend
    allow_credentials=True,
    allow_methods=["*"],  # Allows GET, POST, OPTIONS, etc.
    allow_headers=["*"],  # Allows all headers
)

# Path for FAISS index
faiss_index_path = "faiss_index"

# Initialize embeddings using Google Gemini API
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

class URLInput(BaseModel):
    urls: list[str]

class QueryInput(BaseModel):
    query: str

@app.post("/process-urls")
async def process_urls(url_input: URLInput):
    """Processes the provided URLs, extracts text, splits into chunks, and saves a FAISS index."""
    urls = [url.strip() for url in url_input.urls if url.strip()]
    
    if not urls:
        raise HTTPException(status_code=400, detail="Please provide at least one valid URL.")

    try:
        # Delete existing FAISS index (if any)
        if os.path.exists(faiss_index_path):
            shutil.rmtree(faiss_index_path)

        # Load data from URLs
        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()

        # Split documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
        docs = text_splitter.split_documents(data)

        # Create FAISS index and save it
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(faiss_index_path)

        return {"message": "Documents processed and FAISS index created successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing URLs: {str(e)}")

@app.post("/query")
async def query_docs(query_input: QueryInput):
    """Handles user queries by retrieving answers from the FAISS index using Gemini."""
    if not os.path.exists(faiss_index_path):
        raise HTTPException(status_code=400, detail="No FAISS index found. Please load URLs first.")

    try:
        # Load FAISS index
        vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)

        # Use Gemini 2.0 Flash for faster responses
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())

        def query_with_rate_limit(chain, query, delay=1):
            """Handles rate limits using exponential backoff."""
            try:
                result = chain({"question": query})
                return result
            except Exception as e:
                if "ResourceExhausted: 429" in str(e):
                    time.sleep(delay)
                    return query_with_rate_limit(chain, query, delay * 2)
                else:
                    raise e

        result = query_with_rate_limit(chain, query_input.query)
        
        return {
            "answer": result.get("answer", "No answer found."),
            "sources": result.get("sources", "").strip()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# Run the backend using:
# uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
