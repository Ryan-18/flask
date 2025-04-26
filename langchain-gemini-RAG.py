import warnings
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

from flask import Flask, request, jsonify
from flask_cors import CORS

warnings.filterwarnings("ignore")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS

# Load API keys securely from environment variables
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY environment variable")

# Initialize LangChain model and embeddings
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
    convert_system_message_to_human=True
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

# Load and preprocess documents
loader = TextLoader("Druglist.txt", encoding="utf-8")
documents = loader.load()

# Prepare the context
context = "\n\n".join(str(doc.page_content) for doc in documents)

# Text splitting
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
texts = text_splitter.split_text(context)

# Create a vector database retriever
vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k": 5})

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    model,
    retriever=vector_index,
    return_source_documents=True
)

# API endpoint
@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.json
        question = data.get('question', '')
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        # Query the QA chain
        result = qa_chain({"query": question})
        return jsonify({'answer': result["result"]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Start the server
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
