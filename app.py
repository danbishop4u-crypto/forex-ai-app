import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier

# 1. APPLICATION VIEWPORT SETUP
st.set_page_config(page_title="Real-Time AI Matrix", layout="wide")
st.title("⚡ Real-Time AI Global Market Prediction Dashboard")
st.write("Automatically fetches new data streams and refreshes indicator matrices every 10 seconds.")

# --- REAL-TIME REFRESH MECHANISM CONTROL ---
# Automatically re-runs the entire Python script file on a background timer loop
st.logo("https://icons8.com")
time_delay = 10  # Seconds between automated page updates

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

analysis_vault = {} 

# 3. ADVANCED 8-INDICATOR AI MACHINE LEARNING ENGINE
def run_ai_engine(ticker, interval, period):
    try:
        # Note: We bypass cached data functions here to guarantee a real-time live download stream
        data = yf.download(tickers=ticker, period=period, interval=interval, group_by='ticker', progress=False)
        if data.empty or len(data) < 50:
            return "N/A (Data Error)", 0.0, 0.00000
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(0)
            
        df = pd.DataFrame(data)
        df['Close'] = df['Close'].squeeze()
        df['High'] = df['High'].squeeze()
        df['Low'] = df['Low'].squeeze()
        
        # --- NATIVE MULTI-INDICATOR FEATURE CALCULATIONS ---
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

        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14 + 1e-10))
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()

        tp = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = tp.rolling(window=20).mean()
        mad_tp = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        df['CCI'] = (tp - sma_tp) / (0.015 * mad_tp + 1e-10)

        up_move = df['High'].diff()
        down_move = df['Low'].shift(1) - df['Low']
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
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
        
        if interval == "5m" and ticker not in analysis_vault:
            analysis_vault[ticker] = {
                "rsi": df['RSI'].iloc[-1],
                "macd": df['MACD_Line'].iloc[-1],
                "macd_sig": df['MACD_Signal'].iloc[-1],
                "upper_band": df['BB_Upper'].iloc[-1],
                "lower_band": df['BB_Lower'].iloc[-1],
                "stoch_k": df['Stoch_K'].iloc[-1],
                "cci": df['CCI'].iloc[-1],
                "adx": df['ADX'].iloc[-1],
                "support": df['Low'].tail(25).min(),
                "resistance": df['High'].tail(25).max()
            }
            
        signal = "🚀 BUY" if prediction == 1 else "🩸 SELL"
        return f"{signal} ({accuracy * 100:.0f}% Acc)", accuracy, df['Close'].iloc[-1]
    except Exception as e:
        return "Error", 0.0, 0.00000

# 4. LIVE VIEW CONTAINER RENDER ENGINE
if not selected_assets:
    st.warning("Please check at least one asset box in the left sidebar menu to compute market data.")
else:
    # Visible alert banner flashing the auto update schedule status
    st.info(f"🔄 **Live Feedback Mode Enabled**: Feed updates dynamically every {time_delay} seconds. No manual clicks required.")
    
    matrix_data = []
    analysis_vault.clear() 
    
    with st.spinner("Streaming data updates down from API grids..."):
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
    
    st.subheader("⚡ Live Advanced AI Technical Signal Matrix")
    st.dataframe(result_df, use_container_width=True, hide_index=True)

    # 5. EXPANDED MULTI-INDICATOR STRUCTURAL BREAKDOWN PROFILE
    st.markdown("---")
    st.subheader("📊 Live Indicator Diagnostics (5-Minute Base Chart Profile)")
    
    for asset in selected_assets:
        if asset in analysis_vault:
            metrics = analysis_vault[asset]
            macd_status = "📈 Bullish Cross" if metrics["macd"] > metrics["macd_sig"] else "📉 Bearish Cross"
            adx_status = "🏋️ Strong Trend" if metrics["adx"] > 25 else "💤 Weak / Choppy"
            
            with st.expander(f"👁️ Real-Time Profile: {display_names[asset]}"):
                c1, c2, c3 = st.columns(3)
                
                with c1:
                    st.markdown("**... Channels & Ranges**")
                    st.write(f"* Upper BB: `{metrics['upper_band']:.5f}`")
                    st.write(f"* Lower BB: `{metrics['lower_band']:.5f}`")
                    st.write(f"* ADX Track: `{metrics['adx']:.1f}` ({adx_status})")
                
                with c2:
                    st.markdown("**⚡ Velocity Waves**")
                    st.write(f"* MACD Cross: `{macd_status}`")
                    st.write(f"* Stochastic: `{metrics['stoch_k']:.1f}`")
                    st.write(f"* CCI Track: `{metrics['cci']:.1f}`")
                    
                with c3:
                    st.markdown("**🚧 Static Floors/Ceilings**")
                    st.write(f"* RSI (14P): `{metrics['rsi']:.1f}`")
                    st.write(f"* Local Floor: `{metrics['support']:.5f}`")
                    st.write(f"* Local Ceiling: `{metrics['resistance']:.5f}`")

    # --- TIME TRIGGER ELEMENT LOOP ---
    # Pauses thread briefly before executing page reload instruction
    time.sleep(time_delay)
    st.rerun()
