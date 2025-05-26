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
    print("✅ 進入 handle_message")  # <== 新增這一行
    user_text = event.message.text

    # Step 1: 查找相關內容
    context = search_similar_documents(user_text)

    # Step 2: 組合 prompt 並發送到 Mistral
    prompt = f"根據以下資料回答問題：\n{context}\n問題：{user_text}"
    response = mistral_client.chat.complete(
        model="mistral-small-latest",
        messages=[
            {"role": "system", "content": "你是一位營養專家，請根據資料準確回答。"},
            {"role": "user", "content": prompt}
        ]
    )

    ai_reply = response.choices[0].message.content

    # Step 3: 回覆用戶
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=ai_reply)
    )

if __name__ == "__main__":
    app.run(port=5000)