import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. APPLICATION VIEWPORT SETUP
st.set_page_config(page_title="AI Multi-Asset Scanner", layout="wide")
st.title("🎛️ AI Global Market Multi-Pair & Multi-Timeframe Dashboard")
st.write("Simultaneously processes machine learning trend predictions across worldwide asset networks.")

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

# 3. AUTOMATED LOGIC PREDICTIVE COMPUTATION ENGINE
def run_ai_engine(ticker, interval, period):
    try:
        data = yf.download(tickers=ticker, period=period, interval=interval, group_by='ticker', progress=False)
        if data.empty or len(data) < 35:
            return "N/A (Data Error)", 0.0, 0.00000
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(0)
            
        df = pd.DataFrame(data)
        df['Close'] = df['Close'].squeeze()
        
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['RSI'] = 100 - (100 / (1 + rs))
        
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
        
        signal = "🚀 BUY" if prediction == 1 else "🩸 SELL"
        return f"{signal} ({accuracy * 100:.0f}% Acc)", accuracy, df['Close'].iloc[-1]
    except Exception as e:
        return "Error", 0.0, 0.00000

# 4. TOP REFRESH BUTTON ACTION
if st.button("🔄 Refresh Data (Top)", key="btn_top", use_container_width=True):
    st.cache_data.clear()
    st.toast("Fetching latest live market candles...", icon="⚡")

# 5. APPLICATION MATRIX DISPLAY
if not selected_assets:
    st.warning("Please check at least one asset box in the left sidebar menu to compute market data.")
else:
    matrix_data = []
    
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

    # 6. BOTTOM REFRESH BUTTON ACTION
    st.markdown("---")
    if st.button("🔄 Refresh Data (Bottom)", key="btn_bottom", use_container_width=True):
        st.cache_data.clear()
        st.toast("Fetching latest live market candles...", icon="⚡")
