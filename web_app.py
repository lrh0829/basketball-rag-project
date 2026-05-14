from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from basketball_agent import BasketballRuleAgent

load_dotenv()

app = Flask(__name__)
CORS(app)

API_KEY = os.getenv("API_KEY")

agent = BasketballRuleAgent()

chat_histories = {}

if not API_KEY:
    print("警告: 未找到 API_KEY 环境变量")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        question = data.get('question', '')
        session_id = data.get('session_id', 'default')

        if not question:
            return jsonify({'error': '问题不能为空'}), 400

        if not API_KEY:
            return jsonify({
                'answer': '抱歉，API密钥未配置。请在系统环境变量中设置 API_KEY。',
                'sources': [],
                'error': None
            })

        if session_id not in chat_histories:
            chat_histories[session_id] = []

        history = chat_histories[session_id]

        history.append({"role": "user", "content": question})

        result = agent.get_answer(question, history=history)
        answer = result["result"]
        sources = result.get("source_documents", [])

        history.append({"role": "assistant", "content": answer})

        return jsonify({
            'answer': answer,
            'sources': sources,
            'error': None
        })

    except Exception as e:
        return jsonify({
            'answer': None,
            'sources': [],
            'error': str(e)
        }), 500

@app.route('/api/reset', methods=['POST'])
def reset():
    try:
        data = request.get_json()
        session_id = data.get('session_id', 'default')
        if session_id in chat_histories:
            chat_histories[session_id] = []
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    try:
        session_id = request.args.get('session_id', 'default')
        history = chat_histories.get(session_id, [])
        return jsonify({'history': history})
    except Exception as e:
        return jsonify({'history': [], 'error': str(e)}), 500

if __name__ == '__main__':
    print("启动篮球规则智能问答助手 Web 版本...")
    print("请访问 http://localhost:5000")
    app.run(debug=True, port=5000)