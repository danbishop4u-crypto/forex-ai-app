import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go

st.set_page_config(page_title="AI Forex Engine", layout="wide")
st.title("🤖 AI Forex Market Analyzer")

st.sidebar.header("Parameters")
ticker_input = st.sidebar.selectbox("Currency Pair", ["EURUSD=X", "GBPUSD=X", "USDJPY=X"])
time_period = st.sidebar.selectbox("Lookback", ["60d", "1mo", "3mo"])
time_interval = st.sidebar.selectbox("Interval", ["5m", "15m", "1h", "1d"])

@st.cache_data(ttl=60)
def load_forex_data(ticker, period, interval):
    data = yf.download(tickers=ticker, period=period, interval=interval, group_by='ticker')
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(0)
    return pd.DataFrame(data)

df = load_forex_data(ticker_input, time_period, time_interval)

if df.empty or len(df) < 30:
    st.error("Data error. Try a larger lookback period in the sidebar.")
else:
    # Standardize column structures
    df['Close'] = df['Close'].squeeze()
    df['High'] = df['High'].squeeze()
    df['Low'] = df['Low'].squeeze()
    df['Open'] = df['Open'].squeeze()
    
    # CALCULATE INDICATORS NATIVELY (No dependencies needed)
    # 1. Simple Moving Average (SMA)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    
    # 2. Exponential Moving Average (EMA)
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # 3. Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10) # avoid division by zero
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Establish Target Direction Setup
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df_ml = df.dropna().copy()
    
    features = ['RSI', 'SMA_20', 'EMA_50']
    X = df_ml[features]
    y = df_ml['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    prediction = model.predict(X.iloc[[-1]])
    
    col1, col2 = st.columns(2)
    col1.metric("Live Price", f"{df['Close'].iloc[-1]:.5f}")
    col2.metric("Model Accuracy", f"{accuracy * 100:.1f}%")
    
    if prediction == 1:
        st.success("🤖 AI SIGNAL: BULLISH (BUY TARGET)")
    else:
        st.error("🤖 AI SIGNAL: BEARISH (SELL TARGET)")
        
    st.info("App successfully updated and running smoothly.")
