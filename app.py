import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
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
    # force group_by='ticker' to avoid multi-index errors
    data = yf.download(tickers=ticker, period=period, interval=interval, group_by='ticker')
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(0)
    return pd.DataFrame(data)

df = load_forex_data(ticker_input, time_period, time_interval)

if df.empty or len(df) < 30:
    st.error("Data error. Try a larger lookback period in the sidebar.")
else:
    # Ensure columns are 1D arrays
    df['Close'] = df['Close'].squeeze()
    df['High'] = df['High'].squeeze()
    df['Low'] = df['Low'].squeeze()
    df['Open'] = df['Open'].squeeze()
    
    # Calculate indicators securely
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['SMA_20'] = ta.sma(df['Close'], length=20)
    df['EMA_50'] = ta.ema(df['Close'], length=50)
    
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
    
    if prediction[0] == 1:
        st.success("🤖 AI SIGNAL: BULLISH (BUY TARGET)")
    else:
        st.error("🤖 AI SIGNAL: BEARISH (SELL TARGET)")
