import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 1. APPLICATION VIEWPORT SETUP
st.set_page_config(page_title="AI Multi-Asset Scanner", layout="wide")
st.title("🎛️ AI Global Market Multi-Pair & Multi-Timeframe Dashboard")
st.write("Simultaneously processes machine learning trend predictions and structural market mechanics.")

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

timeframes = {
    "5 Minutes": {"interval": "5m", "period": "5d"},
    "15 Minutes": {"interval": "15m", "period": "14d"},
    "1 Hour": {"interval": "1h", "period": "60d"},
    "1 Day": {"interval": "1d", "period": "1y"}
}

# 3. GLOBAL CORE ANALYTICS COMPUTATION CONTAINER
analysis_vault = {} # Safely houses generated math stats for the summary block below

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
        
        # Core Technical Calculations
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        
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
            
        features = ['RSI', 'SMA_20', 'EMA_50']
        X = df_ml[features]
        y = df_ml['Target']
        
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        model = RandomForestClassifier(n_estimators=30, random_state=42)
        model.fit(X_train, y_train)
        
        accuracy = model.score(X_test, y_test)
        prediction = model.predict(X.iloc[[-1]])
        
        # EXTRACT LIVE VOLATILITY AND STRUCTURE (FOR 1-HOUR STANDARDIZATION)
        if interval == "1h" and ticker not in analysis_vault:
            # Average True Range (ATR) approximation for volatility tracking
            high_low = df['High'] - df['Low']
            atr_sim = high_low.rolling(14).mean().iloc[-1]
            pct_vol = (atr_sim / df['Close'].iloc[-1]) * 100
            
            analysis_vault[ticker] = {
                "rsi": df['RSI'].iloc[-1],
                "sma20": df['SMA_20'].iloc[-1],
                "ema50": df['EMA_50'].iloc[-1],
                "support": df['Low'].tail(30).min(),
                "resistance": df['High'].tail(30).max(),
                "volatility": "🚨 HIGH" if pct_vol > 0.15 else "🟢 LOW" if pct_vol < 0.05 else "📊 MEDIUM"
            }
            
        signal = "🚀 BUY" if prediction == 1 else "🩸 SELL"
        return f"{signal} ({accuracy * 100:.0f}% Acc)", accuracy, df['Close'].iloc[-1]
    except Exception as e:
        return "Error", 0.0, 0.00000

# 4. TOP REFRESH BUTTON ACTION
if st.button("🔄 Refresh Data (Top)", key="btn_top", use_container_width=True):
    st.cache_data.clear()
    analysis_vault.clear() # clear analytics
    st.toast("Fetching latest live market candles...", icon="⚡")

# 5. APPLICATION MATRIX DISPLAY
if not selected_assets:
    st.warning("Please check at least one asset box in the left sidebar menu to compute market data.")
else:
    matrix_data = []
    analysis_vault.clear() # clear storage cache before fresh iteration loops
    
    with st.spinner("Processing AI probability algorithms across global markets..."):
        for asset in selected_assets:
            row = {"Asset Symbol": display_names[asset]}
            latest_price = 0.00000
            
            for tf_name, tf_params in timeframes.items():
                result_str, acc, price = run_ai_engine(asset, tf_params["interval"], tf_params["period"])
                row[tf_name] = result_str
                if price > 0:
                    latest_price = price
                    
            row["Live Market Price"] = f"{latest_price:.5f}" if latest_price > 0 else "Offline"
            matrix_data.append(row)
            
    result_df = pd.DataFrame(matrix_data)
    cols_order = ["Asset Symbol", "Live Market Price", "5 Minutes", "15 Minutes", "1 Hour", "1 Day"]
    result_df = result_df[cols_order]
    
    st.subheader("Live Global Market AI Engine Dashboard Matrix")
    st.dataframe(result_df, use_container_width=True, hide_index=True)

    # 6. EXPANDED STRUCTURAL MARKET ANALYSIS MODULE
    st.markdown("---")
    st.subheader("📊 Live Market Structural Analysis Breakdown (1-Hour Chart Profile)")
    
    for asset in selected_assets:
        if asset in analysis_vault:
            metrics = analysis_vault[asset]
            price_now = float([m["Live Market Price"] for m in matrix_data if m["Asset Symbol"] == display_names[asset]][0])
            
            # Formulate mathematical market bias
            if metrics["rsi"] > 65:
                bias = "🔥 STRONGLY OVERBOUGHT (Look for reversal / exhaustion)"
            elif metrics["rsi"] < 35:
                bias = "❄️ STRONGLY OVERSOLD (Look for support bounces)"
            elif price_now > metrics["ema50"]:
                bias = "📈 MACRO BULLISH CONTINUATION (Price tracking above 50 EMA)"
            else:
                bias = "📉 MACRO BEARISH CONTINUATION (Price tracking below 50 EMA)"
                
            # Render individual clean interface grid blocks for mobile reading
            with st.expander(f"👁️ Analytical Deep-Dive: {display_names[asset]}"):
                c1, c2, c3 = st.columns(3)
                
                with c1:
                    st.markdown("**📊 Current Market Bias**")
                    st.write(bias)
                    st.markdown(f"**⚡ Volatility Signature:** `{metrics['volatility']}`")
                
                with c2:
                    st.markdown("**🚧 Key Structural Horizons**")
                    st.markdown(f"* **Ceiling (Resistance Zone):** `{metrics['resistance']:.5f}`")
                    st.markdown(f"* **Floor (Support Zone):** `{metrics['support']:.5f}`")
                    
                with c3:
                    st.markdown("**📈 Pure Oscillator Telemetry**")
                    st.write(f"* **RSI (14 Period):** `{metrics['rsi']:.1f}`")
                    st.write(f"* **Short Trend (20 SMA):** `{metrics['sma20']:.5f}`")
                    st.write(f"* **Base Trend (50 EMA):** `{metrics['ema50']:.5f}`")

    # 7. BOTTOM REFRESH BUTTON ACTION
    st.markdown("---")
    if st.button("🔄 Refresh Data (Bottom)", key="btn_bottom", use_container_width=True):
        st.cache_data.clear()
        analysis_vault.clear()
        st.toast("Fetching latest live market candles...", icon="⚡")
