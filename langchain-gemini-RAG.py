import warnings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langchain_core.prompts import PromptTemplate
#from langchain import PromptTemplate


from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader

warnings.filterwarnings("ignore")

from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app)  # Enable

# Initialize LangChain model and embeddings
GOOGLE_API_KEY = 'AIzaSyC2FSfa_0xywJmvt-q7gLkEn-Y2EZCObsM'
apii = 'AIzaSyC2FSfa_0xywJmvt-q7gLkEn-Y2EZCObsM'

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
    convert_system_message_to_human=True
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=apii
)

# Load and preprocess documents
loader = TextLoader("Druglist.txt", encoding="utf-8") 
documents = loader.load()
context = "\n\n".join(str(doc.page_content) for doc in documents)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
texts = text_splitter.split_text(context)
vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k": 5})

qa_chain = RetrievalQA.from_chain_type(
    model,
    retriever=vector_index,
    return_source_documents=True
)

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.json
        question = data.get('question', '')
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        # Generate response using LangChain
        result = qa_chain({"query": question})
        return jsonify({'answer': result["result"]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
