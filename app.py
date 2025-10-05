import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from datetime import date
import requests
from textblob import TextBlob

# Page Configuration
st.set_page_config(page_title="Stock Trend Prediction", layout="wide")

# Custom CSS Styling
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='title'>📈 Stock Trend Prediction Dashboard</h1>", unsafe_allow_html=True)

# Dark/Light Mode
dark_mode = st.checkbox("Dark Mode", value=False)
if dark_mode:
    st.markdown('<style>body{background-color:#181818;color:white;}</style>', unsafe_allow_html=True)
else:
    st.markdown('<style>body{background-color:white;color:black;}</style>', unsafe_allow_html=True)

# User Inputs
col1, col2 = st.columns(2)
with col1:
    tickers = st.text_input("Enter Stock Ticker Symbols (e.g., AAPL, GOOG, MSFT)", "AAPL").split(",")
with col2:
    predict_days = st.slider("Days Ahead to Predict", 7, 60, 30)

# Date Range Selector
start_date = st.date_input("Start Date", date(2015, 1, 1))
end_date = st.date_input("End Date", date(2024, 12, 31))

# Model Selection
model_choice = st.selectbox("Select Prediction Model", ["Linear Regression", "Random Forest", "ARIMA"])

# Optional: Enter your NewsAPI key
news_api_key = st.text_input("Enter your NewsAPI Key (optional)", type="password")

# Download and Process Each Stock
for ticker in tickers:
    ticker = ticker.strip()
    st.markdown(f"---\n## 📊 Analysis for {ticker.upper()}")

    df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty:
        st.error(f"❌ Failed to load data for {ticker}. Please check the ticker symbol.")
        continue

    # Show Latest Data
    st.subheader("Latest Stock Data")
    st.dataframe(df.tail())

    # Add Moving Averages
    df['SMA30'] = df['Close'].rolling(window=30).mean()
    df['SMA100'] = df['Close'].rolling(window=100).mean()

    # Plot Price with MAs
    st.subheader("📈 Price Chart with Moving Averages")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA30'], name='30-day SMA'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA100'], name='100-day SMA'))
    st.plotly_chart(fig, use_container_width=True)

    # Candlestick Chart
    st.subheader("🕯️ Candlestick Chart")
    candle_fig = go.Figure(data=[go.Candlestick(x=df.index,
                                                open=df['Open'],
                                                high=df['High'],
                                                low=df['Low'],
                                                close=df['Close'])])
    candle_fig.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(candle_fig, use_container_width=True)

    # Prepare Prediction Data
    df['Prediction'] = df['Close'].shift(-predict_days)
    X = np.array(df[['Close']])[:-predict_days]
    y = np.array(df['Prediction'])[:-predict_days]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Model Training
    if model_choice == "Linear Regression":
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
    elif model_choice == "Random Forest":
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
    elif model_choice == "ARIMA":
        from statsmodels.tsa.arima.model import ARIMA
        arima_model = ARIMA(df['Close'], order=(5, 1, 0))
        arima_result = arima_model.fit()
        predictions = arima_result.forecast(steps=len(y_test))
        predictions = predictions.to_numpy() if hasattr(predictions, "to_numpy") else predictions

    # Evaluation Metrics
    st.subheader("📈 Model Performance")
    st.markdown(f"**R² Score:** {r2_score(y_test, predictions):.4f}")
    st.markdown(f"**MAE:** {mean_absolute_error(y_test, predictions):.2f}")
    st.markdown(f"**MSE:** {mean_squared_error(y_test, predictions):.2f}")
    st.markdown(f"**RMSE:** {np.sqrt(mean_squared_error(y_test, predictions)):.2f}")
    st.markdown(f"**MAPE:** {np.mean(np.abs((y_test - predictions) / y_test)) * 100:.2f}%")

    # Show Predictions
    st.subheader(f"🔮 {predict_days}-Day Forecast Sample")
    pred_df = pd.DataFrame({
        'Actual': y_test[:10],
        'Predicted': predictions[:10]
    })
    st.table(pred_df)

    # Export CSV
    st.download_button(
        label=f"📥 Download {ticker.upper()} Data as CSV",
        data=df.to_csv().encode('utf-8'),
        file_name=f"{ticker}_stock_data.csv",
        mime='text/csv',
        key=f"download_{ticker}"
    )

    # News API + Sentiment
    if news_api_key:
        st.subheader("📰 Latest News & Sentiment")
        try:
            news_url = f'https://newsapi.org/v2/everything?q={ticker}&apiKey={news_api_key}'
            response = requests.get(news_url)
            news_data = response.json()

            if news_data["status"] == "ok":
                for article in news_data["articles"][:5]:
                    st.markdown(f"**{article['title']}**")
                    st.markdown(f"{article['description']}")
                    sentiment = TextBlob(article["description"]).sentiment.polarity
                    sentiment_text = "🟢 Positive" if sentiment > 0 else "🔴 Negative" if sentiment < 0 else "🟡 Neutral"
                    st.markdown(f"**Sentiment:** {sentiment_text}")
                    st.markdown("---")
            else:
                st.warning("⚠️ Unable to fetch news.")
        except Exception as e:
            st.warning(f"❌ News API error: {e}")
