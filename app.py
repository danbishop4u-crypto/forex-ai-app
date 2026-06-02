import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 1. APPLICATION VIEWPORT SETUP
st.set_page_config(page_title="AI Predictive Matrix", layout="wide")
st.title("🎛️ AI Global Market Advanced Multi-Indicator Prediction Dashboard")
st.write("Leverages 5 technical indicator dimensions to feed the machine learning trend engine.")

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

# Timeframes setup optimized for deep lookbacks
timeframes = {
    "1 Minute": {"interval": "1m", "period": "1d"},
    "2 Minutes": {"interval": "2m", "period": "1d"},
    "5 Minutes": {"interval": "5m", "period": "5d"},
    "1 Hour": {"interval": "1h", "period": "60d"}
}

analysis_vault = {} 

# 3. ADVANCED FIVE-INDICATOR AI MACHINE LEARNING ENGINE
def run_ai_engine(ticker, interval, period):
    try:
        data = yf.download(tickers=ticker, period=period, interval=interval, group_by='ticker', progress=False)
        if data.empty or len(data) < 40:
            return "N/A (Data Error)", 0.0, 0.00000
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(0)
            
        df = pd.DataFrame(data)
        df['Close'] = df['Close'].squeeze()
        df['High'] = df['High'].squeeze()
        df['Low'] = df['Low'].squeeze()
        
        # --- NATIVE MULTI-INDICATOR FEATURE CALCULATIONS ---
        # Indicator 1: Simple Moving Average (SMA)
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        
        # Indicator 2: Exponential Moving Average (EMA)
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        
        # Indicator 3: Relative Strength Index (RSI)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Indicator 4: Moving Average Convergence Divergence (MACD)
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD_Line'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
        
        # Indicator 5: Bollinger Bands (Volatility & Target Envelopes)
        std_dev = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['SMA_20'] + (std_dev * 2)
        df['BB_Lower'] = df['SMA_20'] - (std_dev * 2)
        # ---------------------------------------------------
        
        # Define prediction logic boundaries
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df_ml = df.dropna().copy()
        
        if len(df_ml) < 15:
            return "N/A (Data Error)", 0.0, df['Close'].iloc[-1]
            
        # Expanded Features Feeding Module
        features = ['RSI', 'SMA_20', 'EMA_50', 'MACD_Line', 'MACD_Signal', 'BB_Upper', 'BB_Lower']
        X = df_ml[features]
        y = df_ml['Target']
        
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Random Forest initialized with expanded features metrics
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        accuracy = model.score(X_test, y_test)
        prediction = model.predict(X.iloc[[-1]])
        
        # Map parameters for structural visualization modules
        if interval == "5m" and ticker not in analysis_vault:
            analysis_vault[ticker] = {
                "rsi": df['RSI'].iloc[-1],
                "macd": df['MACD_Line'].iloc[-1],
                "macd_sig": df['MACD_Signal'].iloc[-1],
                "upper_band": df['BB_Upper'].iloc[-1],
                "lower_band": df['BB_Lower'].iloc[-1],
                "support": df['Low'].tail(25).min(),
                "resistance": df['High'].tail(25).max()
            }
            
        signal = "🚀 BUY" if prediction == 1 else "🩸 SELL"
        return f"{signal} ({accuracy * 100:.0f}% Acc)", accuracy, df['Close'].iloc[-1]
    except Exception as e:
        return "Error", 0.0, 0.00000

# 4. TOP REFRESH BUTTON ACTION
if st.button("🔄 Refresh Data (Top)", key="btn_top", use_container_width=True):
    st.cache_data.clear()
    analysis_vault.clear() 
    st.toast("Re-calculating predictions using 5 indicators...", icon="⚡")

# 5. APPLICATION MATRIX DISPLAY
if not selected_assets:
    st.warning("Please check at least one asset box in the left sidebar menu to compute market data.")
else:
    matrix_data = []
    analysis_vault.clear() 
    
    with st.spinner("Processing deep multi-indicator AI models..."):
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
    
    st.subheader("⚡ Advanced AI Technical Signal Matrix")
    st.dataframe(result_df, use_container_width=True, hide_index=True)

    # 6. EXPANDED MULTI-INDICATOR STRUCTURAL BREAKDOWN PROFILE
    st.markdown("---")
    st.subheader("📊 Advanced Indicator Diagnostics (5-Minute Base Chart Profile)")
    
    for asset in selected_assets:
        if asset in analysis_vault:
            metrics = analysis_vault[asset]
            
            # Complex Confluence Indicator Engine Definition
            macd_status = "📈 Bullish Cross" if metrics["macd"] > metrics["macd_sig"] else "📉 Bearish Cross"
            
            with st.expander(f"👁️ Advanced Diagnostics Summary: {display_names[asset]}"):
                c1, c2, c3 = st.columns(3)
                
                with c1:
                    st.markdown("**📉 Moving Average Envelopes**")
                    st.write(f"* **Upper Bollinger Band Ceiling:** `{metrics['upper_band']:.5f}`")
                    st.write(f"* **Lower Bollinger Band Floor:** `{metrics['lower_band']:.5f}`")
                    st.write(f" *Bollinger Bands map instant over-extension levels.*")
                
                with c2:
                    st.markdown("**⚡ Momentum Confluence**")
                    st.write(f"* **MACD Momentum Status:** `{macd_status}`")
                    st.write(f"* **MACD Value:** `{metrics['macd']:.6f}`")
                    st.write(f"* **MACD Signal Line:** `{metrics['macd_sig']:.6f}`")
                    
                with c3:
                    st.markdown("**🚧 Core Oscillation Data**")
                    st.write(f"* **RSI (14 Candles):** `{metrics['rsi']:.1f}`")
                    st.write(f"* **Structural Support Floor:** `{metrics['support']:.5f}`")
                    st.write(f"* **Structural Resistance Ceiling:** `{metrics['resistance']:.5f}`")

    # 7. BOTTOM REFRESH BUTTON ACTION
    st.markdown("---")
    if st.button("🔄 Refresh Data (Bottom)", key="btn_bottom", use_container_width=True):
        st.cache_data.clear()
        analysis_vault.clear()
        st.toast("Re-calculating predictions using 5 indicators...", icon="⚡")
