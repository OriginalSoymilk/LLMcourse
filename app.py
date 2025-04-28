from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, TextMessage, TextSendMessage
import requests
import os

app = Flask(__name__)

line_bot_api = LineBotApi(os.environ.get("LINE_CHANNEL_ACCESS_TOKEN"))
handler = WebhookHandler(os.environ.get("LINE_CHANNEL_SECRET"))
mistral_api_key = os.environ.get("MISTRAL_API_KEY")  # 這樣安全一點，不要硬寫死

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except:
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
    headers = {
        "Authorization": f"Bearer {mistral_api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mistral-7b-instruct",  # 看你用哪個模型
        "messages": [
            {"role": "user", "content": user_message}
        ]
    }
    response = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=data)
    return response.json()['choices'][0]['message']['content']

if __name__ == "__main__":
    app.run(port=5000)
