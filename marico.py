import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import date, timedelta

# Streamlit App Title
st.title("📈 Stock Price Prediction App")

# User Input: Stock Symbol
stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA)", "MARICO.NS").upper()

# User Input: Prediction Days
n_days = st.slider("Select number of days for prediction:", 1, 30, 7)

# Fetch Stock Data
def get_stock_data(symbol, period="2y"):
    stock = yf.Ticker(symbol)
    df = stock.history(period=period)
    return df

if st.button("Predict"):
    # Load stock data
    df = get_stock_data(stock_symbol)
    df["Date"] = df.index
    
    # Plot Historical Data
    st.subheader("📊 Historical Stock Prices")
    fig, ax = plt.subplots()
    ax.plot(df["Date"], df["Close"], label="Close Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    st.pyplot(fig)
    
    # Prepare Data for Prediction (Simple Linear Regression)
    df.reset_index(inplace=True)
    df["Days"] = np.arange(len(df))
    X = df["Days"].values.reshape(-1, 1)
    y = df["Close"].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict Next n Days
    future_days = np.arange(len(df), len(df) + n_days).reshape(-1, 1)
    future_preds = model.predict(future_days)
    future_dates = [df["Date"].iloc[-1] + timedelta(days=i) for i in range(1, n_days + 1)]
    
    # Show Predictions
    st.subheader("📈 Predicted Prices")
    pred_df = pd.DataFrame({"Date": future_dates, "Predicted Price": future_preds})
    st.dataframe(pred_df)
    
    # Plot Predictions
    fig2, ax2 = plt.subplots()
    ax2.plot(df["Date"], df["Close"], label="Actual Prices")
    ax2.plot(future_dates, future_preds, "r--", label="Predicted Prices")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Price (USD)")
    ax2.legend()
    st.pyplot(fig2)
