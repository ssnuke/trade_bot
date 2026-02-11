import streamlit as st
import json
import pandas as pd
import time
import os
from pathlib import Path
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="ZeroHero Dashboard", page_icon="⚡")

# --- CUSTOM CSS FOR "PREMIUM" LOOK ---
st.markdown("""
<style>
    .big-font { font-size: 20px !important; }
    .stMetric {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #333;
    }
    div[data-testid="stDataFrame"] {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parents[2]
DASHBOARD_FILE = BASE_DIR / "data" / "dashboard_data.json"

def load_data():
    if not DASHBOARD_FILE.exists(): return None
    try:
        with open(DASHBOARD_FILE, "r") as f: return json.load(f)
    except: return None

# --- MAIN LAYOUT ---
st.title("⚡ ZeroHero Sniper Dashboard")

placeholder = st.empty()

while True:
    data = load_data()
    
    with placeholder.container():
        if not data:
            st.error("Waiting for Bot Data... Please run 'run_bot.bat'")
            time.sleep(2)
            continue
            
        # 1. TOP METRICS ROW
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("💰 Capital", f"₹{data['equity']:,.0f}", f"{data['pnl']:+.0f}")
        m2.metric("🎯 Target", f"₹{data['target']:,.0f}", f"{((data['equity']/data['target'])*100):.1f}%")
        m3.metric("🔥 Mode", data['active_mode'], data['leverage_mode'])
        m4.metric("📊 Total P&L", f"{data['pnl_pct']:.2f}%", f"₹{data['pnl']:,.0f}")
        
        st.markdown("---")

        # 2. ACTIVE & PAST TRADES
        c1, c2 = st.columns([1, 1])
        
        with c1:
            st.subheader("🟢 Active Positions")
            if data['positions']:
                df_pos = pd.DataFrame(data['positions'])
                # Reorder/Rename for clarity
                df_pos = df_pos[['symbol', 'side', 'leverage', 'entry', 'unrealized_pnl']]
                st.dataframe(df_pos, height=200, use_container_width=True)
            else:
                st.info("No Active Positions. Scanning... 🔍")

        with c2:
            st.subheader("📜 Trade History")
            if data['recent_trades']:
                df_hist = pd.DataFrame(data['recent_trades'])
                # Ensure we show Date/Time cleanly
                # Assuming trade['exit_time'] exists or similar, if not we use what we have
                # Given current bot_v2 export, 'recent_trades' has basic fields.
                # Let's format it.
                st.dataframe(df_hist, height=200, use_container_width=True)
            else:
                st.text("No past trades yet.")

        st.markdown("---")

        # 3. SCANNING LOGS (BOTTOM)
        st.subheader("📡 Live Scanning Feed")
        if 'recent_logs' in data and data['recent_logs']:
            # Join logs with newlines to look like a terminal
            log_text = "\n".join(data['recent_logs'][::-1]) # Reverse to show newest on top
            st.code(log_text, language="text")
        else:
            st.text("Connecting to live feed...")

    time.sleep(1) # Fast refresh
