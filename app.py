import os
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from mistralai import Mistral
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

app = Flask(__name__)

# 初始化 LINE API 和 Mistral API
line_bot_api = LineBotApi(os.environ.get("LINE_CHANNEL_ACCESS_TOKEN"))
handler = WebhookHandler(os.environ.get("LINE_CHANNEL_SECRET"))
api_key = os.environ.get("MISTRAL_API_KEY")

# 在這裡初始化 Mistral 客戶端
mistral_client = Mistral(api_key=api_key)

# 初始化 FAISS 相關組件
try:
    print("載入 FAISS 索引和相關資源...")
    # 載入 FAISS 索引
    index = faiss.read_index("nutrition_index.faiss")
    # 載入 documents 對應資料
    with open("nutrition_docs.pkl", "rb") as f:
        documents = pickle.load(f)
    # 載入模型（跟建立階段一樣）
    model = SentenceTransformer('all-MiniLM-L6-v2')
    FAISS_AVAILABLE = True
    print("FAISS 資源載入成功")
except Exception as e:
    print(f"FAISS 資源載入失敗: {e}")
    FAISS_AVAILABLE = False
    index = None
    documents = None
    model = None

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature.")
        abort(400)
    
    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_text = event.message.text
    print(f"收到訊息: {user_text}")
    
    # 如果 FAISS 可用，使用增強版回覆；否則使用原版
    if FAISS_AVAILABLE:
        ai_reply = ask_mistral_with_context(user_text)
    else:
        ai_reply = ask_mistral(user_text)
    
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=ai_reply)
    )

def ask_mistral(user_message):
    # 原版的 Mistral 調用（保持不變）
    chat_response = mistral_client.chat.complete(
        model="mistral-large-latest",
        messages=[{
            "role": "user",
            "content": user_message,
        }]
    )
    return chat_response.choices[0].message.content

def ask_mistral_with_context(user_message):
    # 帶有 FAISS 上下文的增強版
    try:
        print("搜尋相關文檔...")
        # 使用者查詢
        query_embedding = model.encode([user_message])
        D, I = index.search(np.array(query_embedding), k=10)
        
        # 擷取語意最接近的資料
        relevant_docs = [documents[i] for i in I[0]]
        context = "\n".join(relevant_docs)
        
        if context and len(context.strip()) > 0:
            print("找到相關內容，使用上下文回答")
            # 組裝提示詞
            prompt = f"根據以下資料回答問題：\n{context}\n問題：{user_message}"
            
            # 發送給 Mistral AI
            chat_response = mistral_client.chat.complete(
                model="mistral-large-latest",
                messages=[
                    {"role": "system", "content": "你是一位營養專家。請直接根據以下資料回答問題，若資料中有提及，請給出明確的數值，不要猜測或自行補充。資料如下："},
                    {"role": "user", "content": prompt}
                ]
            )
        else:
            print("未找到相關內容，使用一般回答")
            # 如果沒找到相關內容，降級到原版
            chat_response = mistral_client.chat.complete(
                model="mistral-large-latest",
                messages=[{
                    "role": "user",
                    "content": user_message,
                }]
            )
        
        return chat_response.choices[0].message.content
        
    except Exception as e:
        print(f"FAISS 搜尋失敗: {e}")
        # 如果 FAISS 出錯，降級到原版
        return ask_mistral(user_message)

if __name__ == "__main__":
    app.run(port=5000)