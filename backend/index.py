# api/index.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import UnstructuredURLLoader
import os, shutil, time, requests, certifi

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Temporarily open for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SSL Fix
requests.packages.urllib3.util.ssl_.DEFAULT_CA_BUNDLE_PATH = certifi.where()
os.environ['SSL_CERT_FILE'] = certifi.where()

# Load your Google Gemini API Key from env
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
faiss_index_path = "faiss_index"
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

class URLInput(BaseModel):
    urls: list[str]

class QueryInput(BaseModel):
    query: str

@app.post("/process-urls")
async def process_urls(url_input: URLInput):
    urls = [url.strip() for url in url_input.urls if url.strip()]
    if not urls:
        raise HTTPException(status_code=400, detail="No valid URLs provided.")

    try:
        if os.path.exists(faiss_index_path):
            shutil.rmtree(faiss_index_path)

        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
        docs = text_splitter.split_documents(data)

        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(faiss_index_path)
        return {"message": "FAISS index created successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_docs(query_input: QueryInput):
    if not os.path.exists(faiss_index_path):
        raise HTTPException(status_code=400, detail="No FAISS index found.")

    try:
        vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())

        def query_with_rate_limit(chain, query, delay=1):
            try:
                return chain({"question": query})
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
        raise HTTPException(status_code=500, detail=str(e))
