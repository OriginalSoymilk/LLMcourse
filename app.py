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
mistral_client = Mistral(api_key=api_key)

# 載入 FAISS 索引與文件（只載入一次）
index = faiss.read_index("nutrition_index.faiss")
with open("nutrition_docs.pkl", "rb") as f:
    documents = pickle.load(f)

# 載入句子編碼模型
model = SentenceTransformer('all-MiniLM-L6-v2')

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
    ai_reply = ask_mistral_with_faiss(user_text)

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=ai_reply)
    )

def ask_mistral_with_faiss(query):
    # 查詢向量
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k=10)

    # 取得最相關的文件
    relevant_docs = [documents[i] for i in I[0]]
    context = "\n".join(relevant_docs)

    # 組合提示詞
    prompt = f"根據以下資料回答問題：\n{context}\n問題：{query}"

    chat_response = mistral_client.chat.complete(
        model="mistral-large-latest",
        messages=[
            {"role": "system", "content": "你是一位營養專家。請直接根據以下資料回答問題，若資料中有提及，請給出明確的數值，不要猜測或自行補充。資料如下："},
            {"role": "user", "content": prompt}
        ]
    )

    return chat_response.choices[0].message.content

if __name__ == "__main__":
    app.run(port=5000)
