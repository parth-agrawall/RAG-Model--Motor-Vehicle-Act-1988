import os
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from  dotenv import load_dotenv

# Install required libraries
# Run the following command before executing the script:
# !pip install google-generativeai langchain faiss-cpu sentence-transformers python-dotenv

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Gemini LLM
# Use GoogleGenerativeAI instead of ChatGoogleGenerativeAI
llm = GoogleGenerativeAI(
    model="gemini-pro",
    temperature=0.7,
    max_output_tokens=2000,
    google_api_key=GOOGLE_API_KEY,
)

# Load and process text data
def load_and_process_text(file_path):
    loader = TextLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    return texts

# Initialize FAISS vector store with Hugging Face embeddings
def initialize_vector_store(texts):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore

# Query function
def query_rag(vectorstore, query):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    # Use 'stuff' chain type, which is compatible with GoogleGenerativeAI
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

    response = qa_chain(query)

    # Extract sources
    sources = set(doc.metadata["source"] for doc in response["source_documents"] if "source" in doc.metadata)

    return response["result"], sources

# Main function
def main():
    file_path = "motor.txt"  # Replace with your text file path
    texts = load_and_process_text(file_path)
    vectorstore = initialize_vector_store(texts)

    while True:
        query = input("Enter your query: ")
        if query.lower() == "exit":
            break

        if vectorstore is None:
            print("Vector store is not initialized. Please run the script again.")
            continue

        result, sources = query_rag(vectorstore, query)
        print("\nGenerated Response:")
        print(result)
        print("\nReferences:")
        for source in sources:
            print(source)

if __name__ == "__main__":
    main()