import os
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
from functools import lru_cache
from contextlib import asynccontextmanager
import langdetect  # Added for language detection

# MongoDB imports
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch

# Load environment variables if dotenv is available (for local development)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, continue without it

# Configuration from environment variables
MONGODB_URI = os.environ.get("MONGODB_URI")
if not MONGODB_URI:
    raise ValueError("MONGODB_URI environment variable is not set")

DB_NAME = os.environ.get("DB_NAME", "legal_advisor")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "motor_vector_store")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")
    
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TEXT_FILE_PATH = os.environ.get("TEXT_FILE_PATH", "./motor.txt")
DEBUG = os.environ.get("DEBUG", "False").lower() == "true"

# Define lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load MongoDB vector store
    print("Connecting to MongoDB Atlas Vector Search...")
    get_vectorstore()
    yield
    # Shutdown: cleanup could go here if needed

# Initialize FastAPI app with lifespan
app = FastAPI(lifespan=lifespan, title="Multilingual Legal Advisor API", debug=DEBUG)

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Get MongoDB client (cached)
@lru_cache(maxsize=1)
def get_mongo_client():
    """Create and cache the MongoDB client to avoid reinitializing"""
    return MongoClient(
        MONGODB_URI,
        connect=False,  # Defer connection until first operation
        serverSelectionTimeoutMS=60000,
        connectTimeoutMS=60000,
        socketTimeoutMS=60000,
        tls=True,
        tlsAllowInvalidCertificates=True,
        tlsAllowInvalidHostnames=True
    )

# Get embeddings model (cached)
@lru_cache(maxsize=1)
def get_embeddings():
    """Create and cache the embeddings model to avoid reinitializing"""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Initialize Gemini API (cached)
@lru_cache(maxsize=1)
def get_gemini_model():
    """Create and cache the Gemini model to avoid reinitializing"""
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel("gemini-2.0-flash")

# Detect language of text
def detect_language(text: str) -> str:
    """
    Detect the language of the input text.
    Returns the language code (e.g., 'en', 'es', 'fr', etc.)
    """
    try:
        return langdetect.detect(text)
    except:
        # Default to English if detection fails
        return "en"

# Translate text using Gemini model
def translate_text(text: str, target_language: str) -> str:
    """
    Translate text to the specified target language using Gemini LLM.
    """
    gemini_model = get_gemini_model()

    # If target is English
    if target_language == "en":
        translation_prompt = f"""
        Translate the following text to English. Preserve the meaning and technical/legal terms.

        Text: {text}

        Translation:
        """
    # If target is not English
    else:
        translation_prompt = f"""
        Translate the following text to {target_language}. Preserve the meaning and technical/legal terms.

        Text: {text}

        Translation:
        """

    response = gemini_model.generate_content(translation_prompt)
    return response.text.strip()

# Load and split text
def load_and_split_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100
    )
    documents = [Document(page_content=chunk) for chunk in text_splitter.split_text(text)]
    return documents

# Create MongoDB vector store
def create_mongodb_vectorstore(documents):
    client = get_mongo_client()
    embeddings = get_embeddings()
    
    # Create collection with vector search index if it doesn't exist
    db = client[DB_NAME]
    
    # Check if collection exists and drop it if needed for fresh start
    if COLLECTION_NAME in db.list_collection_names():
        db[COLLECTION_NAME].drop()
    
    # Create vector store
    vectorstore = MongoDBAtlasVectorSearch.from_documents(
        documents=documents,
        embedding=embeddings,
        collection=db[COLLECTION_NAME],
        index_name="vector_index",
    )
    
    return vectorstore

# Check if MongoDB collection exists and has documents
def mongodb_collection_exists():
    client = get_mongo_client()
    db = client[DB_NAME]
    
    if COLLECTION_NAME not in db.list_collection_names():
        return False
    
    # Check if collection has documents
    count = db[COLLECTION_NAME].count_documents({})
    return count > 0

# Get MongoDB vector store
def get_mongodb_vectorstore():
    client = get_mongo_client()
    db = client[DB_NAME]
    embeddings = get_embeddings()
    
    return MongoDBAtlasVectorSearch(
        collection=db[COLLECTION_NAME],
        embedding=embeddings,
        index_name="vector_index",
    )

# Load or create MongoDB vector store
def load_or_create_vectorstore(txt_file):
    if not mongodb_collection_exists():
        print("Creating MongoDB Vector Store...")
        documents = load_and_split_text(txt_file)
        vectorstore = create_mongodb_vectorstore(documents)
    else:
        print("Loading MongoDB Vector Store...")
        vectorstore = get_mongodb_vectorstore()
    return vectorstore

# Initialize vector store as a global variable
vectorstore = None

def get_vectorstore():
    global vectorstore
    if vectorstore is None:
        vectorstore = load_or_create_vectorstore(TEXT_FILE_PATH)
    return vectorstore

# Serve the HTML file
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Handle user queries with language support
@app.post("/query")
async def handle_query(query: str = Form(...)):
    try:
        # Detect the language of the query
        source_language = detect_language(query)
        print(f"Detected language: {source_language}")

        # Translate query to English if not already in English
        if source_language != "en":
            english_query = translate_text(query, "en")
            print(f"Translated query: {english_query}")
        else:
            english_query = query

        # Get the vectorstore
        vs = get_vectorstore()

        # Perform similarity search
        docs = vs.similarity_search(english_query, k=4)  # Limit to top 4 relevant documents

        # Prepare context from retrieved documents
        context = " ".join(doc.page_content for doc in docs)

        # Generate response using Gemini API
        gemini_model = get_gemini_model()
        prompt = f"""You are a legal advisor. You are assisting the user to better understand Indian Laws and Regulation. 

Question: {english_query}

Guide them with their queries and take references from this Context: {context}. 
Don't mention about the context or that you have any prior knowledge. 
Don't write phrases like 'based on the information available' or 'based on your information'."""

        response = gemini_model.generate_content(prompt)
        english_answer = response.text

        # Translate the answer back to the original language if needed
        if source_language != "en":
            translated_answer = translate_text(english_answer, source_language)
            result = translated_answer
            print(f"Translated response to {source_language}")
        else:
            result = english_answer

        # Return the response
        return JSONResponse(content={"result": result, "sources": []})

    except Exception as e:
        error_message = f"An error occurred while processing your query: {str(e)}"

        # Try to translate error message if language was detected
        try:
            if source_language and source_language != "en":
                translated_error = translate_text(error_message, source_language)
                return JSONResponse(content={"result": translated_error, "sources": []})
        except:
            pass

        return JSONResponse(content={"result": error_message, "sources": []})

# Add a healthcheck endpoint for monitoring
@app.get("/health")
async def health_check():
    try:
        # Check if we can connect to MongoDB
        client = get_mongo_client()
        client.admin.command('ping')
        
        # Check if we can get the vectorstore
        vs = get_vectorstore()
        
        return {"status": "healthy"}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e)}
        )

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    
    # Use PORT environment variable if set (for Render)
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=DEBUG)
