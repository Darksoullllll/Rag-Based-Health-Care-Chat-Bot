from flask import Flask, render_template, request, jsonify
from langchain.embeddings import HuggingFaceEmbeddings  #type:ignore
from api_keys import PINECONE_API, GEMINI_API
from langchain_pinecone import PineconeVectorStore  #type:ignore
from dotenv import load_dotenv
import google.generativeai as genai  #type:ignore
from langchain.chains import create_retrieval_chain #type:ignore
import os
import tempfile

app = Flask(__name__)

# set environment variables for api
os.environ["PINECONE_API_KEY"] = PINECONE_API
os.environ["GEMINI_API_KEY"] = GEMINI_API

# Load HuggingFace embeddings
def download_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings

embeddings = download_embeddings()

# Load Pinecone VectorStore
index_name = "aibot"
vector = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = vector.as_retriever(search_type='similarity', search_kwargs={'k': 10})

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# system prompt
system_prompt = (
    "You are a medical assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say that you don't know. "
    "Provide a concise answer and possible treatment."
    "\n\n"
    "{context}"
)

system_prompt_img = """
Medical Image Analysis Responsibilities
As a highly skilled medical practitioner specializing in image analysis, your expertise is crucial in examining medical images for a renowned hospital. Your role includes identifying anomalies, diseases, or health issues present in the images.

Responsibilities:
1. Thorough Image Analysis:
   - Carefully analyze each image, focusing on any significant findings that may indicate health issues.

2. Findings Report:
   - Document all observed anomalies or signs of disease.
   - Clearly articulate these findings in a structured format for easy comprehension.

3. Recommendations and Next Steps:
   - Based on your analysis, suggest potential next steps, including further tests or treatments as applicable.

4. Treatment Suggestions:
   - If appropriate, recommend possible treatment options or interventions.

Important Notes:
1. Scope of Response:
   - Only respond if the image pertains to human health issues.

2. Clarity of Image:
   - In cases where image quality impedes clear analysis, note that certain aspects cannot be determined based on the provided image.

3. Disclaimer:
   - Accompany your analysis with the following disclaimer: "Consult with a Doctor before making any decisions."
4. Your Insights are invaluable in guiding clinical decision. please procceed with the analysis,adhering to the structured approach outlined above
"""

# Configure the Gemini model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
)
def format_text(response):
    formatted_response = response.replace("**", "").replace("*", "-")
    return formatted_response

# Helper function to call Gemini API
def call_gemini_api(formatted_input):
    user_input = formatted_input["input"]
    context = formatted_input["context"]

    # Start chat session
    chat_session = model.start_chat(
        history=[
            {"role": "model", "parts": [{"text": system_prompt.format(context=context)}]},
            {"role": "user", "parts": [{"text": user_input}]},
        ]
    )

    # Send message and return response
    response = chat_session.send_message({"parts": [{"text": user_input}]})
    return response.text

# img based chat-bot
def upload_genai(file_path, mime_type=None):
    file = genai.upload_file(file_path, mime_type=mime_type)
    return file

# img process
def process_img(files):
    
    chat_session = model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [
                        files[0],
                    ],
                }
            ]
        )

    response = chat_session.send_message(system_prompt_img)
    
    return response

# Flask route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for the chat interface
@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/chat-image')
def img():
    return render_template('image-chat.html')

# API route to handle chat messages
@app.route('/api/chat', methods=['POST'])
def api_chat():
    user_message = request.json['message']

    # Retrieve context documents based on user input
    context_documents = retriever.invoke(user_message)
    context_text = "\n\n".join([doc.page_content for doc in context_documents])

    # Format input for the Gemini API
    formatted_input = {
        "input": user_message,
        "context": context_text
    }

    # Call the Gemini API to get the response
    response = call_gemini_api(formatted_input)
    
    formatted_response = format_text(response)

    # Return response to the frontend
    return jsonify({"response": formatted_response})

@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    image = request.files['image']

    # Save the image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(image.read())
        temp_file_path = temp_file.name

    try:
        # Upload the image using genai
        files = [upload_genai(temp_file_path, mime_type=image.mimetype)]
        
        # Process the image using AI model
        diagnosis = process_img(files)

        return jsonify({'response': diagnosis.text})

    finally:
        os.remove(temp_file_path)


if __name__ == '__main__':
    app.run(debug=True)
