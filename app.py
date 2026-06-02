import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 1. APPLICATION VIEWPORT SETUP
st.set_page_config(page_title="AI Hyper-Scalper Engine", layout="wide")
st.title("⚡ AI Global Market Hyper-Scalper Multi-Timeframe Dashboard")
st.write("Simultaneously processes real-time machine learning predictions across micro-scalping intervals.")

# 2. EXPANDED SELECTION ASSET INVENTORY 
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

# HYPER-SCALPING TIMEFRAME INTERVALS (Requires a maximum of 30 days lookback for 1m-5m charts)
timeframes = {
    "1 Minute": {"interval": "1m", "period": "1d"},
    "2 Minutes": {"interval": "2m", "period": "1d"},
    "3 Minutes": {"interval": "3m", "period": "1d"},
    "5 Minutes": {"interval": "5m", "period": "5d"}
}

# 3. GLOBAL CORE ANALYTICS COMPUTATION CONTAINER
analysis_vault = {} 

def run_ai_engine(ticker, interval, period):
    try:
        data = yf.download(tickers=ticker, period=period, interval=interval, group_by='ticker', progress=False)
        if data.empty or len(data) < 35:
            return "N/A (Data Error)", 0.0, 0.00000
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(0)
            
        df = pd.DataFrame(data)
        df['Close'] = df['Close'].squeeze()
        df['High'] = df['High'].squeeze()
        df['Low'] = df['Low'].squeeze()
        
        # Fast Technical Calculations for Scalping
        df['SMA_10'] = df['Close'].rolling(window=10).mean() # Faster trend line
        df['EMA_25'] = df['Close'].ewm(span=25, adjust=False).mean() # Faster base line
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # ML Engine Target Allocation
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df_ml = df.dropna().copy()
        
        if len(df_ml) < 10:
            return "N/A (Data Error)", 0.0, df['Close'].iloc[-1]
            
        features = ['RSI', 'SMA_10', 'EMA_25']
        X = df_ml[features]
        y = df_ml['Target']
        
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        model = RandomForestClassifier(n_estimators=35, random_state=42)
        model.fit(X_train, y_train)
        
        accuracy = model.score(X_test, y_test)
        prediction = model.predict(X.iloc[[-1]])
        
        # EXTRACT LIVE MICRO-VOLATILITY AND STRUCTURE (FOR 5-MINUTE CHART)
        if interval == "5m" and ticker not in analysis_vault:
            high_low = df['High'] - df['Low']
            atr_sim = high_low.rolling(14).mean().iloc[-1]
            pct_vol = (atr_sim / df['Close'].iloc[-1]) * 100
            
            analysis_vault[ticker] = {
                "rsi": df['RSI'].iloc[-1],
                "sma10": df['SMA_10'].iloc[-1],
                "ema25": df['EMA_25'].iloc[-1],
                "support": df['Low'].tail(20).min(), # Recent 20-candle floor
                "resistance": df['High'].tail(20).max(), # Recent 20-candle ceiling
                "volatility": "🚨 HIGH (Fast Moves)" if pct_vol > 0.08 else "🟢 LOW (Calm Market)" if pct_vol < 0.03 else "📊 MEDIUM"
            }
            
        signal = "🚀 BUY" if prediction == 1 else "🩸 SELL"
        return f"{signal} ({accuracy * 100:.0f}% Acc)", accuracy, df['Close'].iloc[-1]
    except Exception as e:
        return "Error", 0.0, 0.00000

# 4. TOP REFRESH BUTTON ACTION
if st.button("🔄 Refresh Scalper Data (Top)", key="btn_top", use_container_width=True):
    st.cache_data.clear()
    analysis_vault.clear() 
    st.toast("Fetching latest live price ticks...", icon="⚡")

# 5. APPLICATION MATRIX DISPLAY
if not selected_assets:
    st.warning("Please check at least one asset box in the left sidebar menu to compute market data.")
else:
    matrix_data = []
    analysis_vault.clear() 
    
    with st.spinner("Processing AI scalp patterns across micro-intervals..."):
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
    cols_order = ["Asset Symbol", "Live Price", "1 Minute", "2 Minutes", "3 Minutes", "5 Minutes"]
    result_df = result_df[cols_order]
    
    st.subheader("⚡ Live Hyper-Scalping AI Signal Matrix")
    st.dataframe(result_df, use_container_width=True, hide_index=True)

    # 6. EXPANDED MICRO STRUCTURAL MARKET ANALYSIS MODULE
    st.markdown("---")
    st.subheader("📊 Scalping Structural Profile (5-Minute Base Chart)")
    
    for asset in selected_assets:
        if asset in analysis_vault:
            metrics = analysis_vault[asset]
            
            try:
                price_now = float([m["Live Price"] for m in matrix_data if m["Asset Symbol"] == display_names[asset]][0])
            except:
                price_now = 0.0
                
            if metrics["rsi"] > 70:
                bias = "🔥 SCALP OVERBOUGHT (Extreme momentum, high risk to buy)"
            elif metrics["rsi"] < 30:
                bias = "❄️ SCALP OVERSOLD (Extreme oversold, look for immediate long scalps)"
            elif price_now > metrics["ema25"]:
                bias = "📈 MICRO BULLISH BIAS (Price trading above fast 25 EMA)"
            else:
                bias = "📉 MICRO BEARISH BIAS (Price trading below fast 25 EMA)"
                
            with st.expander(f"👁️ Scalping Dashboard Summary: {display_names[asset]}"):
                c1, c2, c3 = st.columns(3)
                
                with c1:
                    st.markdown("**📊 Execution Bias**")
                    st.write(bias)
                    st.markdown(f"**⚡ Micro Volatility:** `{metrics['volatility']}`")
                
                with c2:
                    st.markdown("**🚧 Scalping Targets**")
                    st.markdown(f"* **Immediate Ceiling (Resistance):** `{metrics['resistance']:.5f}`")
                    st.markdown(f"* **Immediate Floor (Support):** `{metrics['support']:.5f}`")
                    
                with c3:
                    st.markdown("**📈 Scalper Indicators**")
                    st.write(f"* **Fast RSI (14 Candles):** `{metrics['rsi']:.1f}`")
                    st.write(f"* **Micro Trend (10 SMA):** `{metrics['sma10']:.5f}`")
                    st.write(f"* **Base Scalp (25 EMA):** `{metrics['ema25']:.5f}`")

    # 7. BOTTOM REFRESH BUTTON ACTION
    st.markdown("---")
    if st.button("🔄 Refresh Scalper Data (Bottom)", key="btn_bottom", use_container_width=True):
        st.cache_data.clear()
        analysis_vault.clear()
        st.toast("Fetching latest live price ticks...", icon="⚡")
