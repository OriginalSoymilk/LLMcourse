import os
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from mistralai import Mistral

app = Flask(__name__)

line_bot_api = LineBotApi(os.environ.get("LINE_CHANNEL_ACCESS_TOKEN"))
handler = WebhookHandler(os.environ.get("LINE_CHANNEL_SECRET"))
mistral_client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))

# 安全的導入 query_faiss 模組
try:
    from query_faiss import search_similar_documents
    FAISS_AVAILABLE = True
    print("FAISS 模組載入成功")
except ImportError as e:
    print(f"FAISS 模組載入失敗: {e}")
    FAISS_AVAILABLE = False
except Exception as e:
    print(f"FAISS 模組載入時發生錯誤: {e}")
    FAISS_AVAILABLE = False

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    print(f"收到 webhook 請求: {body[:100]}...")  # 只印前100字符

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature.")
        abort(400)
    except Exception as e:
        print(f"處理 webhook 時發生錯誤: {e}")
        abort(500)

    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    try:
        user_text = event.message.text
        print(f"收到 LINE 訊息：{user_text}")
        
        # 檢查 FAISS 是否可用
        if FAISS_AVAILABLE:
            try:
                # Step 1: 查找相關內容
                context = search_similar_documents(user_text)
                print(f"找到相關內容，長度: {len(context) if context else 0}")
                
                # Step 2: 組合 prompt 並發送到 Mistral
                prompt = f"根據以下資料回答問題：\n{context}\n問題：{user_text}"
            except Exception as e:
                print(f"FAISS 搜尋失敗: {e}")
                # 降級到簡單模式
                prompt = user_text
        else:
            # FAISS 不可用時的降級處理
            print("FAISS 不可用，使用簡單模式")
            prompt = user_text

        # Step 3: 調用 Mistral API
        response = mistral_client.chat.complete(
            model="mistral-small-latest",
            messages=[
                {"role": "system", "content": "你是一位營養專家，請根據資料準確回答。"},
                {"role": "user", "content": prompt}
            ]
        )

        ai_reply = response.choices[0].message.content
        print(f"AI 回覆: {ai_reply[:50]}...")  # 只印前50字符

        # Step 4: 回覆用戶
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=ai_reply)
        )
        print("訊息回覆成功")

    except Exception as e:
        print(f"處理訊息時發生錯誤: {e}")
        # 發送錯誤訊息給用戶
        try:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="抱歉，處理您的訊息時發生錯誤，請稍後再試。")
            )
        except Exception as reply_error:
            print(f"回覆錯誤訊息失敗: {reply_error}")

# 添加健康檢查端點
@app.route("/health", methods=['GET'])
def health_check():
    return {"status": "healthy", "faiss_available": FAISS_AVAILABLE}

# 添加根路由
@app.route("/", methods=['GET'])
def home():
    return "LINE Bot is running!"

if __name__ == "__main__":
    print("啟動 Flask 應用...")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)