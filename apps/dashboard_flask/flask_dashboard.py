from flask import Flask, render_template, jsonify, send_from_directory
import json
import os
from pathlib import Path
import requests

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
DASHBOARD_FILE = BASE_DIR / "data" / "dashboard_data.json"
BOT_API_URL = os.getenv("BOT_API_URL", "").strip()

def load_data():
    if BOT_API_URL:
        try:
            url = f"{BOT_API_URL.rstrip('/')}/dashboard_data"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching bot data: {e}")

    if not DASHBOARD_FILE.exists():
        print(f"File not found: {DASHBOARD_FILE}")
        return None
    try:
        with open(DASHBOARD_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading dashboard data: {e}")
        return None

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/data')
def get_data():
    data = load_data()
    if data:
        return jsonify(data)
    return jsonify({"error": "No data available"}), 404

@app.route('/api/session_history')
def get_session_history():
    history_dir = BASE_DIR / "apps" / "bot" / "paper_trades" / "sessions"
    if not history_dir.exists():
        return jsonify([])
    
    history = []
    try:
        for file in history_dir.glob("session_*.json"):
            with open(file, 'r') as f:
                history.append(json.load(f))
        
        # Sort by date descending
        history.sort(key=lambda x: x.get('date', ''), reverse=True)
        return jsonify(history)
    except Exception as e:
        print(f"Error loading session history: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", "5006"))
    print("🚀 Premium Delta Bot Dashboard starting...")
    print(f"👉 URL: http://localhost:{port}")
    print(f"📡 Serving data from: {DASHBOARD_FILE}")
    app.run(host='0.0.0.0', port=port, debug=False)
