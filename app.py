import os
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from mistralai import Mistral
from query_faiss import search_similar_documents

app = Flask(__name__)

line_bot_api = LineBotApi(os.environ.get("LINE_CHANNEL_ACCESS_TOKEN"))
handler = WebhookHandler(os.environ.get("LINE_CHANNEL_SECRET"))
mistral_client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))
@app.route("/", methods=["GET"])
def index():
    return "✅ Flask server is running", 200

@app.route("/callback", methods=['POST'])
def callback():
    print("📩 收到 LINE webhook")  # <== 新增這一行
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
    try:
        print("✅ 進入 handle_message")
        user_text = event.message.text
        print("👤 使用者輸入：", user_text)

        context = search_similar_documents(user_text)

        prompt = f"根據以下資料回答問題：\n{context}\n問題：{user_text}"
        response = mistral_client.chat.complete(
            model="mistral-small-latest",
            messages=[
                {"role": "system", "content": "你是一位營養專家，請根據資料準確回答。"},
                {"role": "user", "content": prompt}
            ]
        )

        ai_reply = response.choices[0].message.content

        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=ai_reply)
        )
    except Exception as e:
        print("❌ handle_message 發生錯誤：", e)
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="伺服器出現錯誤，請稍後再試 🙇")
        )

if __name__ == "__main__":
    app.run(port=5000)