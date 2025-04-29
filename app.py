from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embedding, format_docs
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.prompt import template
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os 

load_dotenv()

app = Flask(__name__)

os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

embeddings = download_hugging_face_embedding()
index_name = "medical-chat-bot"
vector_db = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
retriever = vector_db.as_retriever(search_type='similarity', search_kwargs={'k': 4})

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
prompt = PromptTemplate.from_template(template=template)
parser = StrOutputParser()

chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()}
         | prompt
         | llm
         | parser)

@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/get", methods=['POST'])
def chat():
    data = request.get_json()
    question = data.get("question", "")
    
    if not question:
        return jsonify({"response": "Please enter a valid question."})

    print('User question:', question)
    bot_response = chain.invoke(question)
    print('Chatbot response:', bot_response)

    return jsonify({"response": bot_response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
