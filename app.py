import os
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from mistralai import Mistral

app = Flask(__name__)

# 初始化 LINE API 和 Mistral API
line_bot_api = LineBotApi(os.environ.get("LINE_CHANNEL_ACCESS_TOKEN"))
handler = WebhookHandler(os.environ.get("LINE_CHANNEL_SECRET"))
api_key = os.environ.get("MISTRAL_API_KEY")

# 在這裡初始化 Mistral 客戶端
mistral_client = Mistral(api_key=api_key)

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
    ai_reply = ask_mistral(user_text)

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=ai_reply)
    )

def ask_mistral(user_message):
    # 使用 mistralai 库呼叫 Mistral API
    chat_response = mistral_client.chat.complete(
        model="mistral-large-latest",  # 使用你需要的模型
        messages=[{
            "role": "user",
            "content": user_message,
        }]
    )

    # 返回 AI 的回應內容
    return chat_response.choices[0].message.content

if __name__ == "__main__":
    app.run(port=5000)