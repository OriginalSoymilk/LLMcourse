import os
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from mistralai import Mistral # Corrected: This should be MistralClient
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# --- Configuration and Global Initializations ---
app = Flask(__name__)

# LINE API Configuration
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET")
if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_CHANNEL_SECRET:
    print("Error: LINE Channel Access Token or Secret not found in environment variables.")
    # Consider exiting or raising an exception if these are critical for startup
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# Mistral AI Configuration
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    print("Error: MISTRAL_API_KEY not found in environment variables.")
    # Consider exiting or raising an exception
mistral_client = Mistral(api_key=MISTRAL_API_KEY) # This was Mistral(), should be MistralClient if using newer versions, or stick to Mistral if that's the class name in your mistralai lib version

# RAG (Retrieval Augmented Generation) Components
FAISS_INDEX_PATH = "nutrition_index.faiss"
DOCUMENTS_PATH = "nutrition_docs.pkl"
SENTENCE_MODEL_NAME = 'all-MiniLM-L6-v2' # A good balance of size and performance
N_RETRIEVED_DOCS = 3 # Number of documents to retrieve, keep this low to manage prompt size and memory

try:
    print(f"Loading SentenceTransformer model: {SENTENCE_MODEL_NAME}...")
    sentence_model = SentenceTransformer(SENTENCE_MODEL_NAME)
    print("SentenceTransformer model loaded.")

    print(f"Loading FAISS index from: {FAISS_INDEX_PATH}...")
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    print("FAISS index loaded.")

    print(f"Loading documents from: {DOCUMENTS_PATH}...")
    with open(DOCUMENTS_PATH, "rb") as f:
        documents = pickle.load(f)
    print("Documents loaded.")
    print(f"Number of documents: {len(documents)}")
    print(f"FAISS index entries: {faiss_index.ntotal}")
    if len(documents) != faiss_index.ntotal:
        print("Warning: Mismatch between number of documents and FAISS index entries. Ensure they correspond.")

except FileNotFoundError as e:
    print(f"Error loading RAG files: {e}. Make sure '{FAISS_INDEX_PATH}' and '{DOCUMENTS_PATH}' exist.")
    # Depending on your desired behavior, you might want to exit or disable RAG functionality
    sentence_model = None
    faiss_index = None
    documents = None
    print("RAG functionality will be disabled.")
except Exception as e:
    print(f"An unexpected error occurred during RAG component loading: {e}")
    sentence_model = None
    faiss_index = None
    documents = None
    print("RAG functionality will be disabled.")


# --- Flask Routes ---
@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    app.logger.info(f"Request body: {body}") # Good for debugging

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel secret and access token.")
        abort(400)
    except Exception as e:
        app.logger.error(f"Error in callback: {e}")
        abort(500)
    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_text = event.message.text
    ai_reply = "I am having trouble processing your request right now." # Default reply

    # Use the RAG system if components are loaded
    if sentence_model and faiss_index and documents:
        try:
            ai_reply = ask_nutrition_expert_with_rag(user_text)
        except Exception as e:
            app.logger.error(f"Error in RAG processing: {e}")
            # Fallback or specific error message
            ai_reply = "Sorry, I encountered an issue looking up nutrition information."
    else:
        # Fallback to general Mistral chat if RAG components failed to load
        # Or, you could decide to only offer nutrition advice and inform the user
        # that the nutrition service is currently unavailable.
        # For this example, let's provide a message indicating service limitation.
        app.logger.warning("RAG components not loaded. Nutrition specific queries might not work optimally.")
        ai_reply = "The nutrition information service is currently initializing or unavailable. Please try again later."
        # Optionally, you could fall back to a general Mistral query without RAG here if desired:
        # ai_reply = ask_general_mistral(user_text)


    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=ai_reply)
    )

# --- Helper Functions ---
def ask_nutrition_expert_with_rag(user_query: str) -> str:
    """
    Answers a user query using RAG with FAISS, SentenceTransformers, and Mistral AI.
    """
    if not sentence_model or not faiss_index or not documents:
        return "Nutrition knowledge base is not available."

    print(f"RAG: Encoding query: '{user_query}'")
    query_embedding = sentence_model.encode([user_query])

    print(f"RAG: Searching FAISS index (k={N_RETRIEVED_DOCS})...")
    # D: distances, I: indices of nearest neighbors
    D, I = faiss_index.search(np.array(query_embedding), k=N_RETRIEVED_DOCS)

    relevant_docs_content = []
    if I.size > 0 and I[0][0] != -1: # Check if any results and not invalid index
        for i in I[0]:
            if 0 <= i < len(documents):
                relevant_docs_content.append(documents[i])
            else:
                app.logger.warning(f"RAG: Invalid index {i} from FAISS search results.")
    
    if not relevant_docs_content:
        context = "No specific information found in the knowledge base for this query."
        print("RAG: No relevant documents found in FAISS.")
    else:
        context = "\n".join(relevant_docs_content)
        print(f"RAG: Retrieved context: {context[:200]}...") # Log snippet of context

    # Assemble the prompt for Mistral AI
    prompt = f"根據以下資料回答問題：\n{context}\n\n問題：{user_query}"

    print("RAG: Sending prompt to Mistral AI...")
    try:
        chat_response = mistral_client.chat.complete(
            model="mistral-large-latest", # Or your preferred Mistral model
            messages=[
                {"role": "system", "content": "你是一位營養專家。請直接根據提供的資料回答問題。如果資料中沒有明確提及，請說明資料不足。請給出明確的數值（若資料中有）。不要猜測或自行補充資料中未提及的內容。"},
                {"role": "user", "content": prompt}
            ]
        )
        answer = chat_response.choices[0].message.content
        print(f"RAG: Mistral AI response: {answer}")
        return answer
    except Exception as e:
        app.logger.error(f"Mistral API call failed: {e}")
        return "Sorry, I had trouble contacting the AI to answer your question based on the nutrition data."

# Optional: If you want a general Mistral function without RAG as a fallback
# def ask_general_mistral(user_message: str) -> str:
#     """
#     Sends a message to Mistral API for general conversation.
#     """
#     try:
#         chat_response = mistral_client.chat.complete(
#             model="mistral-large-latest",
#             messages=[{"role": "user", "content": user_message}]
#         )
#         return chat_response.choices[0].message.content
#     except Exception as e:
#         app.logger.error(f"General Mistral API call failed: {e}")
#         return "Sorry, I'm having trouble thinking right now."

# --- Main Execution ---
if __name__ == "__main__":
    # Ensure PORT environment variable is set, Render provides this automatically
    port = int(os.environ.get("PORT", 5000))
    # For local development, you might want to enable debug mode.
    # For production on Render, Gunicorn will be used (debug=False is default and recommended)
    app.run(host='0.0.0.0', port=port, debug=False) # Set debug=True for local dev if needed