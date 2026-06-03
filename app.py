import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from streamlit_autorefresh import st_autorefresh

# 1. APPLICATION VIEWPORT SETUP
st.set_page_config(page_title="High-Accuracy AI Predictive Matrix", layout="wide")
st.title("🎛️ Optimized AI Global Market Dashboard & Confirmation Chat")
st.write("Combines high-accuracy technical trend matrices with an on-demand strategy confirmation engine.")

# Safe automated backend updates every 30 seconds
count = st_autorefresh(interval=30000, limit=1000, key="forex_auto_refresh")

# 2. SELECTION ASSET INVENTORY
st.sidebar.header("Asset Grid Controls")

asset_catalog = {
    "🔱 Precious Metals & Crypto": ["GC=F", "SI=F", "BTC-USD"],
    "🔥 Major Currency Pairs": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X"],
    "📈 Minor & Exotic Pairs": ["EURGBP=X", "EURJPY=X", "GBPJPY=X", "USDZAR=X"]
}

display_names = {
    "GC=F": "XAU/USD (Gold Spot)",
    "SI=F": "XAG/USD (Silver Spot)",
    "BTC-USD": "BTC/USD (Bitcoin)",
    "EURUSD=X": "EUR/USD (Euro / Dollar)",
    "GBPUSD=X": "GBP/USD (Pound / Dollar)",
    "USDJPY=X": "USD/JPY (Dollar / Yen)",
    "AUDUSD=X": "AUD/USD (Aussie / Dollar)",
    "USDCAD=X": "USD/CAD (Loonie / Dollar)",
    "USDCHF=X": "USD/CHF (Swiss / Dollar)",
    "EURGBP=X": "EUR/GBP (Euro / Pound)",
    "EURJPY=X": "EUR/JPY (Euro / Yen)",
    "GBPJPY=X": "GBP/JPY (Pound / Yen)",
    "USDZAR=X": "USD/ZAR (Dollar / Rand)"
}

selected_assets = []
for category, items in asset_catalog.items():
    st.sidebar.markdown(f"### {category}")
    for item in items:
        default_checked = (item in ["GC=F", "EURUSD=X"])
        if st.sidebar.checkbox(f"Add {display_names[item]}", value=default_checked, key=f"side_{item}"):
            selected_assets.append(item)

timeframes = {
    "1 Minute": {"interval": "1m", "period": "1d"},
    "2 Minutes": {"interval": "2m", "period": "1d"},
    "5 Minutes": {"interval": "5m", "period": "5d"},
    "1 Hour": {"interval": "1h", "period": "60d"}
}

# Persistent cross-refresh session state allocation containers
if "analysis_vault" not in st.session_state:
    st.session_state.analysis_vault = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 3. HIGH-ACCURACY AI MACHINE LEARNING ENGINE
def run_ai_engine(ticker, interval, period):
    try:
        data = yf.download(tickers=ticker, period=period, interval=interval, group_by='ticker', progress=False)
        if data.empty or len(data) < 55:
            return "N/A (Data Error)", 0.0, 0.00000
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(0)
            
        df = pd.DataFrame(data)
        df['Close'] = df['Close'].squeeze()
        df['High'] = df['High'].squeeze()
        df['Low'] = df['Low'].squeeze()
        
        # --- TECHNICAL INDICATOR PROCESSING ---
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD_Line'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
        
        std_dev = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['SMA_20'] + (std_dev * 2)
        df['BB_Lower'] = df['SMA_20'] - (std_dev * 2)
        
        # --- STATIONARY FEATURE TRANSFORMATION ---
        df['Dist_SMA20'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
        df['Dist_EMA50'] = (df['Close'] - df['EMA_50']) / df['EMA_50']
        df['Dist_BB_Upper'] = (df['BB_Upper'] - df['Close']) / df['Close']
        df['Dist_BB_Lower'] = (df['Close'] - df['BB_Lower']) / df['Close']
        df['MACD_Diff'] = df['MACD_Line'] - df['MACD_Signal']
        
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df_ml = df.dropna().copy()
        
        if len(df_ml) < 20:
            return "N/A (Data Error)", 0.0, df['Close'].iloc[-1]
            
        features = ['RSI', 'Dist_SMA20', 'Dist_EMA50', 'MACD_Diff', 'Dist_BB_Upper', 'Dist_BB_Lower']
        X = df_ml[features]
        y = df_ml['Target']
        
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        model = RandomForestClassifier(n_estimators=150, min_samples_split=5, random_state=42)
        model.fit(X_train, y_train)
        
        accuracy = model.score(X_test, y_test)
        latest_row = X.iloc[[-1]]
        pred_prob = model.predict_proba(latest_row)
        max_prob = np.max(pred_prob)
        raw_pred = model.predict(latest_row)
        
        if max_prob < 0.53:
            signal = "⏳ NEUTRAL"
        else:
            signal = "🚀 BUY" if raw_pred == 1 else "🩸 SELL"
        
        # Keep internal session database fully populated
        st.session_state.analysis_vault[ticker] = {
            "signal": signal,
            "accuracy": f"{accuracy * 100:.0f}%",
            "rsi": float(df['RSI'].iloc[-1]),
            "macd_cross": "Bullish" if df['MACD_Line'].iloc[-1] > df['MACD_Signal'].iloc[-1] else "Bearish",
            "price": float(df['Close'].iloc[-1]),
            "support": float(df['Low'].tail(25).min()),
            "resistance": float(df['High'].tail(25).max())
        }
            
        return f"{signal} ({accuracy * 100:.0f}% Acc)", accuracy, df['Close'].iloc[-1]
    except Exception as e:
        return "Error", 0.0, 0.00000

# 4. APPLICATION MATRIX DISPLAY
if not selected_assets:
    st.warning("Please check at least one asset box in the left sidebar menu to compute market data.")
else:
    matrix_data = []
    
    with st.spinner("Syncing latest live data streams..."):
        for asset in selected_assets:
            row = {"Asset Symbol": display_names[asset]}
            latest_price = 0.00000
            
            for tf_name, tf_params in timeframes.items():
                result_str, acc, price = run_ai_engine(asset, tf_params["interval"], tf_params["period"])
                row[tf_name] = result_str
                if price > 0:
                    latest_price = price
                    
            row["Live Price"] = f"{latest_price:.5f}" if latest_price > 0 else "Offline"
            matrix_data.append(row)
            
    result_df = pd.DataFrame(matrix_data)
    cols_order = ["Asset Symbol", "Live Price", "1 Minute", "2 Minutes", "5 Minutes", "1 Hour"]
    result_df = result_df[cols_order]
    
    st.subheader("⚡ Optimized High-Accuracy AI Technical Signal Matrix")
    st.dataframe(result_df, use_container_width=True, hide_index=True)

    # 5. INTEGRATED LIVE AI CONFIRMATION CHAT WINDOW
    st.markdown("---")
    st.subheader("💬 AI Strategy Confirmation Chat Room")
    st.write("Type an asset name (e.g., 'gold', 'bitcoin', 'eurusd') to unlock confirmation logic maps.")

    # Container to hold chat messages so they display reliably in order
    chat_container = st.container()

    # User chat entry interface setup
    user_query = st.chat_input("Verify active market entry parameters...")

    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        bot_response = ""
        matched = False
        q = user_query.lower().replace("/", "").replace(" ", "").replace("-", "")
        
        # Flattened loops to prevent formatting/indentation alignment crashes
        for key, name in display_names.items():
            k_clean = key.lower().replace("=x", "").replace("-", "")
            
            # Pure single-line logic conditions preventing multi-line block breakages
            match_found = (k_clean in q) or ("gold" in q and key == "GC=F") or ("xau" in q and key == "GC=F") or ("silver" in q and key == "SI=F") or ("xag" in q and key == "SI=F") or ("bitcoin" in q and key == "BTC-USD") or ("btc" in q and key == "BTC-USD") or ("eur" in q and "usd" in q and key == "EURUSD=X") or ("gbp" in q and "usd" in q and key == "GBPUSD=X")
            
            if match_found:
                matched = True
                if key in st.session_state.analysis_vault:
                    m = st.session_state.analysis_vault[key]
                    status_text = 'Overbought' if m['rsi'] > 70 else 'Oversold' if m['rsi'] < 30 else 'Neutral Momentum'
                    
                    bot_response = f"📊 **Live Confirmation Blueprint for {name}:**\n\n"
                    bot_response += f"* **AI Direction Trend Recommendation:** `{m['signal']}` (Calculated over 150 optimized baseline indices)\n"
                    bot_response += f"* **Live Execution Quote:** `{m['price']:.5f}`\n"
                    bot_response += f"* **Oscillator Wave Status:** RSI is at `{m['rsi']:.1f}` ({status_text})\n"
                    bot_response += f"* **Momentum Struct:** MACD is showcasing a `{m['macd_cross']}` direction phase.\n\n"
                    bot_response += f"🚧 **Definitive Order Management Bounds:**\n"
                    bot_response += f" * **Invalidation Protective Stop-Loss:** `{m['support']:.5f}`\n"
                    bot_response += f" * **Baseline Scalping Target Take-Profit:** `{m['resistance']:.5f}`"
                else:
                    bot_response = f"Live statistics for {name} are initializing. Verify that the asset checkbox in the left sidebar menu is checked, then submit your query again."
                break
                
        if not matched:
