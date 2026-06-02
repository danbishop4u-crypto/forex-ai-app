import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 1. APPLICATION VIEWPORT SETUP
st.set_page_config(page_title="AI Trend Predictive Matrix", layout="wide")
st.title("🎛️ AI Global Market Ultra-Trend Prediction Dashboard")
st.write("Leverages 8 advanced technical and trend-strength indicator dimensions to feed the machine learning prediction engine.")

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

# Timeframes setup optimized for deep indicator mathematical lag lookbacks
timeframes = {
    "1 Minute": {"interval": "1m", "period": "1d"},
    "2 Minutes": {"interval": "2m", "period": "1d"},
    "5 Minutes": {"interval": "5m", "period": "5d"},
    "1 Hour": {"interval": "1h", "period": "60d"}
}

analysis_vault = {} 

# 3. ADVANCED EIGHT-INDICATOR AI MACHINE LEARNING ENGINE
def run_ai_engine(ticker, interval, period):
    try:
        data = yf.download(tickers=ticker, period=period, interval=interval, group_by='ticker', progress=False)
        if data.empty or len(data) < 50:
            return "N/A (Data Error)", 0.0, 0.00000
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(0)
            
        df = pd.DataFrame(data)
        df['Close'] = df['Close'].squeeze()
        df['High'] = df['High'].squeeze()
        df['Low'] = df['Low'].squeeze()
        
        # --- NATIVE DUAL TREND & OSCILLATOR MACHINE FEATURES ---
        # 1. Moving Averages Base
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        
        # 2. Relative Strength Index (RSI)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 3. MACD
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD_Line'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
        
        # 4. Bollinger Bands
        std_dev = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['SMA_20'] + (std_dev * 2)
        df['BB_Lower'] = df['SMA_20'] - (std_dev * 2)
        
        # 5. Stochastic Oscillator (%K and %D) - Identifies overextended wave momentum
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14 + 1e-10))
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        # 6. Commodity Channel Index (CCI) - Tracks cyclical trend breakouts
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = tp.rolling(window=20).mean()
        mad_tp = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        df['CCI'] = (tp - sma_tp) / (0.015 * mad_tp + 1e-10)
        
        # 7. Average Directional Index (ADX) Approximation - Isolates absolute trend speed/strength
        up_move = df['High'].diff()
        down_move = df['Low'].shift(1) - df['Low']
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Simplified true range calculation
        tr = np.maximum(df['High'] - df['Low'], np.maximum(np.abs(df['High'] - df['Close'].shift(1)), np.abs(df['Low'] - df['Close'].shift(1))))
        atr_14 = pd.Series(tr).rolling(window=14).mean()
        
        plus_di = 100 * (pd.Series(plus_dm).rolling(window=14).mean() / (atr_14 + 1e-10))
        minus_di = 100 * (pd.Series(minus_dm).rolling(window=14).mean() / (atr_14 + 1e-10))
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        df['ADX'] = pd.Series(dx).rolling(window=14).mean().values
        # ---------------------------------------------------
        
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df_ml = df.dropna().copy()
        
        if len(df_ml) < 15:
            return "N/A (Data Error)", 0.0, df['Close'].iloc[-1]
            
        # Expanded Features Matrix feeding the core forest engine
        features = ['RSI', 'SMA_20', 'EMA_50', 'MACD_Line', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'Stoch_K', 'Stoch_D', 'CCI', 'ADX']
        X = df_ml[features]
        y = df_ml['Target']
        
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        accuracy = model.score(X_test, y_test)
        prediction = model.predict(X.iloc[[-1]])
        
        # Safely pass variables downstream into UI container
        if interval == "5m" and ticker not in analysis_vault:
            analysis_vault[ticker] = {
                "rsi": df['RSI'].iloc[-1],
                "macd_status": "📈 Bullish Cross" if df['MACD_Line'].iloc[-1] > df['MACD_Signal'].iloc[-1] else "📉 Bearish Cross",
                "stoch_k": df['Stoch_K'].iloc[-1],
                "stoch_d": df['Stoch_D'].iloc[-1],
                "cci": df['CCI'].iloc[-1],
                "adx": df['ADX'].iloc[-1],
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
    st.toast("Re-calculating predictions using 8 trend indicators...", icon="⚡")

# 5. APPLICATION MATRIX DISPLAY
if not selected_assets:
    st.warning("Please check at least one asset box in the left sidebar menu to compute market data.")
else:
    matrix_data = []
    analysis_vault.clear() 
    
    with st.spinner("Processing deep structural AI trend engines..."):
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

    # 6. EXPANDED MULTI-TREND STRUCTURAL BREAKDOWN PROFILE
    st.markdown("---")
    st.subheader("📊 Advanced Trend Analytics Dashboard (5-Minute Base Chart)")
    
    for asset in selected_assets:
        if asset in analysis_vault:
            metrics = analysis_vault[asset]
            
            # Formulate mathematical interpretation strings for easy mobile tracking
            adx_speed = "🏋️ STRONG TREND" if metrics["adx"] > 25 else "💤 WEAK / CHOPPY RANGE"
            stoch_status = "🔥 Overbought" if metrics["stoch_k"] > 80 else "❄️ Oversold" if metrics["stoch_k"] < 20 else "Neutral"
            cci_status = "🚀 Bullish Breakout" if metrics["cci"] > 100 else "🩸 Bearish Breakout" if metrics["cci"] < -100 else "Consolidating"
            
            with st.expander(f"👁️ Advanced Trend Analytics Summary: {display_names[asset]}"):
                c1, c2, c3 = st.columns(3)
                
                with c1:
                    st.markdown("**🏃 Trend Strength Profile**")
                    st.write(f"* **ADX Trend Velocity:** `{metrics['adx']:.1f}` ➔ **{adx_speed}**")
                    st.write(f"* **CCI Momentum Track:** `{metrics['cci']:.1f}` ➔ **{cci_status}**")
                    st.caption("ADX measures structural trend speed; values over 25 signify sustained momentum.")
                
                with c2:
                    st.markdown("**🌊 Wave Oscillations**")
