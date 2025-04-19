# main1.py
import os
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
from functools import lru_cache
from contextlib import asynccontextmanager
import langdetect  # Added for language detection

os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Define lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load vectorstore
    print("Loading FAISS Vector Store...")
    get_vectorstore()
    yield
    # Shutdown: cleanup could go here if needed


# Initialize FastAPI app with lifespan
app = FastAPI(lifespan=lifespan, title="Multilingual Legal Advisor API")

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Load environment variables or configurations
txt_file = "./motor.txt"
faiss_index_path = "faiss_index"
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"


# Get embeddings model (cached)
@lru_cache(maxsize=1)
def get_embeddings():
    """Create and cache the embeddings model to avoid reinitializing"""
    return HuggingFaceEmbeddings(model_name=embedding_model)


# Initialize Gemini API (cached)
@lru_cache(maxsize=1)
def get_gemini_model():
    """Create and cache the Gemini model to avoid reinitializing"""
    api_key = "AIzaSyAvkXnso2NhNoUIu5Hlu9H3l8HLdI2N5jc"
    genai.configure(api_key=api_key)
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


# Create FAISS vector store
def create_faiss_vectorstore(documents, faiss_index_path):
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(faiss_index_path)
    return vectorstore


# Load FAISS vector store
def load_faiss_vectorstore(faiss_index_path):
    embeddings = get_embeddings()
    return FAISS.load_local(
        faiss_index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )


# Load or create FAISS vector store
def load_or_create_vectorstore(txt_file, faiss_index_path):
    if not os.path.exists(faiss_index_path):
        print("Creating FAISS Vector Store...")
        documents = load_and_split_text(txt_file)
        vectorstore = create_faiss_vectorstore(documents, faiss_index_path)
    else:
        print("Loading FAISS Vector Store...")
        vectorstore = load_faiss_vectorstore(faiss_index_path)
    return vectorstore


# Initialize vector store as a global variable
vectorstore = None


def get_vectorstore():
    global vectorstore
    if vectorstore is None:
        vectorstore = load_or_create_vectorstore(txt_file, faiss_index_path)
    return vectorstore


# Serve the HTML file
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.head("/")
async def head_root():
    return {}  # Return an empty response for HEAD requests
    
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


# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    # Use reload mode but only one worker for simpler startup
    uvicorn.run("main1:app", host="0.0.0.0", port=8001, reload=True)
