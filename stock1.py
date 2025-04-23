import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tempfile
import os
import sys
from io import StringIO
import contextlib
import yfinance as yf
from bs4 import BeautifulSoup
import requests
import math
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from urllib.request import urlopen, Request
from datetime import datetime, date, timedelta
import plotly.graph_objects as go
import matplotlib.dates as mdates
from talib import RSI





# =============================================
# NEW CODE START - Fundamental Analysis Helpers
# =============================================

def get_pe_ratio(ticker):
    """Get P/E ratio from yfinance, fallback to Finviz"""
    try:
        # Try yfinance first
        stock = yf.Ticker(ticker)
        pe = stock.info.get('trailingPE', stock.info.get('forwardPE', None))
        
        if pe is not None:
            return pe
            
        # Fallback to Finviz
        finviz_data = fetch_finviz_data(ticker)
        if finviz_data and finviz_data.get('P/E'):
            return float(finviz_data['P/E'])
            
        return None
    except Exception as e:
        st.error(f"Error getting P/E ratio: {str(e)}")
        return None

def get_roe(ticker):
    """Get ROE from yfinance, fallback to Finviz"""
    try:
        # Try yfinance first
        stock = yf.Ticker(ticker)
        roe = stock.info.get('returnOnEquity', None)
        
        if roe is not None:
            return roe * 100  # Convert to percentage
            
        # Calculate from financials if possible
        try:
            income_stmt = stock.quarterly_financials
            balance_sheet = stock.quarterly_balance_sheet
            
            if 'Net Income' in income_stmt.index and 'Total Stockholder Equity' in balance_sheet.index:
                net_income = income_stmt.loc['Net Income'].iloc[0]
                equity = balance_sheet.loc['Total Stockholder Equity'].iloc[0]
                return (net_income / equity) * 100
        except:
            pass
            
        # Fallback to Finviz
        finviz_data = fetch_finviz_data(ticker)
        if finviz_data and finviz_data.get('ROE'):
            return float(finviz_data['ROE'])
            
        return None
    except Exception as e:
        st.error(f"Error getting ROE: {str(e)}")
        return None

def fetch_finviz_data(ticker):
    """Scrapes key metrics from Finviz with robust error handling"""
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Parse all key-value pairs from the snapshot table
        data = {}
        rows = soup.find_all('tr', class_=['table-dark-row', 'table-light-row'])
        
        for row in rows:
            cols = row.find_all('td')
            if len(cols) == 2:
                key = cols[0].text.strip()
                value = cols[1].text.strip()
                data[key] = value
        
        # Extract and clean specific metrics
        metrics = {
            'P/E': clean_numeric(data.get('P/E')),
            'ROE': clean_numeric(data.get('ROE'), is_percentage=True),
            'Beta': clean_numeric(data.get('Beta'))
        }
        
        return metrics
        
    except Exception as e:
        st.warning(f"Couldn't fetch Finviz data: {str(e)}")
        return None

def clean_numeric(value, is_percentage=False):
    """Cleans and converts string values to numeric"""
    if not value or value == '-':
        return None
    try:
        # Remove percentage signs if present
        if is_percentage:
            value = value.replace('%', '')
        # Remove commas and convert to float
        return float(value.replace(',', ''))
    except:
        return None


def analyze_news_sentiment(ticker):
    """Enhanced sentiment analysis returning compound score"""
    try:
        finviz_url = f'https://finviz.com/quote.ashx?t={ticker}'
        req = Request(url=finviz_url, headers={'user-agent': 'my-app'})
        response = urlopen(req)
        html = BeautifulSoup(response, 'html.parser')
        news_table = html.find(id='news-table')

        # Parse news data
        vader = SentimentIntensityAnalyzer()
        sentiments = []
        
        for row in news_table.findAll('tr'):
            title = row.a.text
            score = vader.polarity_scores(title)['compound']
            sentiments.append(score)
        
        return {
            'compound_mean': np.mean(sentiments) if sentiments else 0,
            'positive': len([s for s in sentiments if s > 0.05]),
            'neutral': len([s for s in sentiments if -0.05 <= s <= 0.05]),
            'negative': len([s for s in sentiments if s < -0.05])
        }
    except Exception as e:
        st.error(f"Sentiment analysis error: {str(e)}")
        return None



def get_stock_price(ticker):
    """Fetches the current stock price using yfinance."""
    stock = yf.Ticker(ticker)
    stock_info = stock.history(period="1d")
    return stock_info['Close'].iloc[-1]

def get_implied_volatility(ticker):
    """Scrapes the 30-day implied volatility from AlphaQuery."""
    url = f'https://www.alphaquery.com/stock/{ticker}/volatility-option-statistics/30-day/iv-mean'
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    rows = soup.find_all('tr')
    for row in rows:
        cols = row.find_all('td')
        if len(cols) == 2 and "Implied Volatility (Mean)" in cols[0].text:
            iv_text = cols[1].text.strip().replace('%', '')
            return float(iv_text) / 100
    return None

def calculate_expected_range(current_price, implied_volatility, days=30):
    """Calculates the expected price fluctuation and range."""
    if implied_volatility is None:
        return None, None, None
    fluctuation = current_price * implied_volatility * math.sqrt(days / 365)
    lower_bound = current_price - fluctuation
    upper_bound = current_price + fluctuation
    return fluctuation, lower_bound, upper_bound

def plot_volatility_analysis(current_price, fluctuation, company):
    """Plots the expected price fluctuation range."""
    today = date.today()
    dates = [today + timedelta(days=i) for i in range(31)]  # 30 days
    price_path = [current_price] * 31
    upper_bound = [current_price + (fluctuation * math.sqrt(i / 365)) for i in range(31)]
    lower_bound = [current_price - (fluctuation * math.sqrt(i / 365)) for i in range(31)]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, price_path, label='Current Price', linestyle='--', color='blue')
    ax.plot(dates, upper_bound, label='Upper Bound (1σ)', color='green')
    ax.plot(dates, lower_bound, label='Lower Bound (1σ)', color='red')
    ax.fill_between(dates, lower_bound, upper_bound, color='gray', alpha=0.4)
    ax.set_title(f"Expected {company} Stock Price Fluctuation Over Next 30 Days")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.legend()
    ax.grid(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%M-%D'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig, upper_bound, lower_bound, price_path

def show_model_comparison(arima_results, lstm_results, test_data, company):
    """Displays model comparison in a new expandable section with robust error handling"""
    with st.expander(f"🔍 {company} MODEL COMPARISON", expanded=True):
        st.header(f"📊 {company} Model Performance Comparison")
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📈 Metrics Comparison", "📉 Predictions Plot", "📌 Model Details"])
        
        with tab1:
            # Metrics comparison table with error handling
            st.subheader("Performance Metrics")
            
            # Safely get metrics with defaults
            arima_rmse = arima_results.get('RMSE', float('nan'))
            lstm_rmse = lstm_results.get('RMSE', float('nan'))
            
            metrics_data = {
                "Metric": ["RMSE", "MAE", "MAPE (%)", "R² Score"],
                "ARIMA": [
                    f"{arima_rmse:.4f}" if not pd.isna(arima_rmse) else "N/A",
                    f"{arima_results.get('MAE', 'N/A'):.4f}" if 'MAE' in arima_results else "N/A",
                    f"{arima_results.get('MAPE', 'N/A'):.2f}" if 'MAPE' in arima_results else "N/A",
                    f"{arima_results.get('R²', 'N/A'):.4f}" if 'R²' in arima_results else "N/A"
                ],
                "LSTM": [
                    f"{lstm_rmse:.4f}" if not pd.isna(lstm_rmse) else "N/A",
                    f"{lstm_results.get('MAE', 'N/A'):.4f}" if 'MAE' in lstm_results else "N/A",
                    f"{lstm_results.get('MAPE', 'N/A'):.2f}" if 'MAPE' in lstm_results else "N/A",
                    f"{lstm_results.get('R²', 'N/A'):.4f}" if 'R²' in lstm_results else "N/A"
                ]
            }
            
            metrics_df = pd.DataFrame(metrics_data)
            
            # Highlight better performing metrics only if both models have values
            def highlight_better(val):
                if val == "N/A":
                    return ''
                try:
                    if "RMSE" in val or "MAE" in val or "MAPE" in val:
                        arima_val = float(arima_results.get(val.split()[0], float('nan')))
                        lstm_val = float(lstm_results.get(val.split()[0], float('nan')))
                        if pd.isna(arima_val) or pd.isna(lstm_val):
                            return ''
                        if arima_val < lstm_val:
                            return 'background-color: #ffcccc'  # Red for ARIMA
                        else:
                            return 'background-color: #ccffcc'  # Green for LSTM
                    elif "R²" in val:
                        arima_r2 = float(arima_results.get('R²', float('nan')))
                        lstm_r2 = float(lstm_results.get('R²', float('nan')))
                        if pd.isna(arima_r2) or pd.isna(lstm_r2):
                            return ''
                        if arima_r2 > lstm_r2:
                            return 'background-color: #ffcccc'
                        else:
                            return 'background-color: #ccffcc'
                except:
                    return ''
                return ''
            
            st.dataframe(
                metrics_df.style.applymap(highlight_better, subset=["ARIMA", "LSTM"]),
                hide_index=True
            )
            
            # Performance summary (only show if we have both RMSE values)
            if not pd.isna(arima_rmse) and not pd.isna(lstm_rmse):
                st.subheader("Summary")
                better_model = "ARIMA" if arima_rmse < lstm_rmse else "LSTM"
                st.success(f"🏆 Best Performing Model: **{better_model}**")
                relative_perf = (min(arima_rmse, lstm_rmse)/max(arima_rmse, lstm_rmse))*100
                diff = abs(arima_rmse-lstm_rmse)/max(arima_rmse, lstm_rmse)*100
                st.metric("Relative Performance", 
                         f"{relative_perf:.1f}%",
                         delta=f"{diff:.1f}% difference")
        
        with tab2:
            # Visual comparison with error handling
            st.subheader("Actual vs Predicted Prices")
            
            fig = go.Figure()
            
            # Add actual prices if available
            if len(test_data) > 0:
                fig.add_trace(go.Scatter(
                    x=test_data.index[-len(arima_results.get('predictions', [])):],
                    y=test_data['Close'].values[-len(arima_results.get('predictions', [])):],
                    name="Actual Prices",
                    line=dict(color='blue', width=2)
                ))
            
            # Add ARIMA predictions if available
            if 'predictions' in arima_results and len(arima_results['predictions']) > 0:
                fig.add_trace(go.Scatter(
                    x=test_data.index[-len(arima_results['predictions']):],
                    y=arima_results['predictions'],
                    name="ARIMA",
                    line=dict(color='red', width=1.5, dash='dot')
                ))
            
            # Add LSTM predictions if available
            if 'predictions' in lstm_results and len(lstm_results['predictions']) > 0:
                fig.add_trace(go.Scatter(
                    x=test_data.index[-len(lstm_results['predictions']):],
                    y=lstm_results['predictions'].flatten(),
                    name="LSTM",
                    line=dict(color='green', width=1.5, dash='dash')
                ))
            
            fig.update_layout(
                title=f"{company} Price Predictions Comparison",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Detailed reports with error handling
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("ARIMA Details")
                if 'model_summary' in arima_results:
                    st.text(arima_results['model_summary'])
                else:
                    st.warning("Model summary not available for ARIMA")
                
                if 'AIC' in arima_results:
                    st.metric("AIC", f"{arima_results['AIC']:.2f}")
                else:
                    st.warning("AIC metric not available")
            
            with col2:
                st.subheader("LSTM Details")
                details = []
                if 'params' in lstm_results:
                    details.append(f"- **Parameters**: {lstm_results['params']:,}")
                else:
                    details.append("- Parameters: N/A")
                
                if 'train_loss' in lstm_results:
                    details.append(f"- **Final Training Loss**: {lstm_results['train_loss']:.4f}")
                
                if 'val_loss' in lstm_results:
                    details.append(f"- **Validation Loss**: {lstm_results['val_loss']:.4f}")
                
                if details:
                    st.markdown("\n".join(details))
                else:
                    st.warning("No training details available")
                
                if 'epochs' in lstm_results:
                    st.metric("Training Epochs", lstm_results['epochs'])


# Set page config
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .sidebar .sidebar-content {
        background-color: #e1f5fe;
    }
    h1 {
        color: #0d47a1;
    }
    h2 {
        color: #1976d2;
    }
    .stButton>button {
        background-color: #0d47a1;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Page title
st.title("📊 Stock Analysis Dashboard")
st.markdown("""
This dashboard provides comprehensive analysis of Apple and Google stock prices using ARIMA and LSTM models.
""")

# Sidebar for navigation
# Company Selection - Dropdown
company_choice = st.selectbox("Select Company for Analysis:", ["Apple", "Google"])

# File Upload - For either Apple or Google
uploaded_file = st.file_uploader(f"Upload {company_choice} stock data (CSV format)", type="csv")


# Function to capture print output
@contextlib.contextmanager
def capture_output():
    new_out = StringIO()
    old_out = sys.stdout
    try:
        sys.stdout = new_out
        yield new_out
    finally:
        sys.stdout = old_out

# Function to run ARIMA analysis
def run_arima_analysis(df, company):
    st.subheader("ARIMA Model Analysis")
    
    # Use only 'Close' price for ARIMA
    stock_prices = df['Close']
    
    # ADF Test
    with st.expander("Augmented Dickey-Fuller Test Results"):
        res = adfuller(stock_prices)
        st.write(f"ADF statistic: {res[0]}")
        st.write(f"p-value: {res[1]}")
    
    # ARIMA model
    order = (8,1,1) if company == "Apple" else (6,1,0)
    model = ARIMA(stock_prices, order=order)
    out = model.fit()
    
    with st.expander("ARIMA Model Summary"):
        st.text(out.summary())
    
    # Make predictions
    pred = out.predict(start=1, end=len(stock_prices)-1)
    
    # Plot actual vs predicted
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df.index, stock_prices, color='red', linestyle='solid', label='Actual Close')
    ax.plot(df.index[1:len(pred)+1], pred, color='blue', label='ARIMA Prediction', linestyle='dotted')
    ax.set_title(f'{company} Stock Price: Actual vs ARIMA Prediction')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    
    # Calculate metrics
    actual = stock_prices.values[1:len(pred)+1]
    predicted = pred.values
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    r2 = r2_score(actual, predicted)
    
    # Display metrics
    metrics_df = pd.DataFrame({
        'Metric': ['MSE', 'RMSE', 'MAE', 'MAPE', 'R²'],
        'Value': [mse, rmse, mae, mape, r2]
    })
    st.dataframe(metrics_df.style.format({'Value': '{:.4f}'}))
    
    # Residual analysis
    residuals = actual - predicted
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_res = plt.figure(figsize=(6,4))
        plt.plot(residuals, label='Residuals')
        plt.axhline(0, color='red', linestyle='--')
        plt.title(f"{company} ARIMA Residual Plot")
        plt.xlabel("Time Step")
        plt.ylabel("Prediction Error")
        plt.legend()
        plt.grid(True)
        st.pyplot(fig_res)
    
    with col2:
        fig_hist = plt.figure(figsize=(6,4))
        plt.hist(residuals, bins=50, edgecolor='black', color='orchid')
        plt.title(f'{company} Histogram of Residuals ARIMA')
        plt.xlabel('Residual Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        st.pyplot(fig_hist)
    
    # Forecast future prices
    forecast_30 = out.forecast(steps=30)
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date+pd.Timedelta(days=1), periods=30)
    
    fig_forecast = plt.figure(figsize=(10,4))
    #plt.plot(df.index, stock_prices, label='Historical Price', color='blue')
    plt.plot(future_dates, forecast_30, label='30-Day Forecast', color='orange', linestyle='--', marker='o')
    plt.title(f"{company} Stock Forecast - Next 30 Days (ARIMA)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    st.pyplot(fig_forecast) 
    return {
        'predictions': predicted,
        'residuals': residuals,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R²': r2,
        'model_summary': out.summary().as_text()
        
    }

# Function to create sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Function to run LSTM analysis
def run_lstm_analysis(df, company):
    st.subheader("LSTM Model Analysis")

    
    # Select the 'Close' price for prediction
    closing_prices = df[['Close']].values
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0,1))
    closing_prices_scaled = scaler.fit_transform(closing_prices)
    
    # Create sequences
    seq_length = 60
    X_all, y_all = create_sequences(closing_prices_scaled, seq_length)
    X_all = X_all.reshape((X_all.shape[0], X_all.shape[1], 1))
    
    # Build the LSTM model
    model = Sequential([
        LSTM(units=128, return_sequences=True, input_shape=(seq_length, 1)),  
        Dropout(0.3),
        LSTM(units=64, return_sequences=True),  
        Dropout(0.3),
        LSTM(units=64),  
        Dropout(0.3),
        Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Split into train and test
    train_size = int(len(closing_prices_scaled) * 0.8)
    train_data = closing_prices_scaled[:train_size]
    test_data = closing_prices_scaled[train_size:]
    X_train, y_train = create_sequences(train_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    # Train the model
    with st.spinner('Training LSTM model...'):
        history = model.fit(X_train, y_train, epochs=20, batch_size=32, 
                          validation_data=(X_test, y_test), verbose=0)
    
    # Plot training loss
    fig_loss = plt.figure(figsize=(10,4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'LSTM Model Training Loss ({company})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    st.pyplot(fig_loss)
    
    # Make predictions
    y_pred_all = model.predict(X_all)
    y_pred_all_inv = scaler.inverse_transform(y_pred_all)
    y_actual_all_inv = scaler.inverse_transform(y_all.reshape(-1, 1))
    
    # Align predictions with timeline
    aligned_preds = np.empty_like(closing_prices)
    aligned_preds[:] = np.nan
    aligned_preds[seq_length:] = y_pred_all_inv
    
    # Plot actual vs predicted
    fig_compare = plt.figure(figsize=(10,4))
    plt.plot(closing_prices, label='Actual Price', color='blue')
    plt.plot(aligned_preds, label='Predicted Price', color='red', linestyle='dashed')
    plt.title(f"{company} Stock Prediction (Full Dataset)")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.legend()
    st.pyplot(fig_compare)
    
    # Calculate metrics
    test_pred = model.predict(X_test)
    test_pred_inv = scaler.inverse_transform(test_pred)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    mse = mean_squared_error(y_test_inv, test_pred_inv)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_inv, test_pred_inv)
    mape = np.mean(np.abs((y_test_inv - test_pred_inv) / y_test_inv)) * 100
    r2 = r2_score(y_test_inv, test_pred_inv)
    
    # Display metrics
    metrics_df = pd.DataFrame({
        'Metric': ['MSE', 'RMSE', 'MAE', 'MAPE', 'R²'],
        'Value': [mse, rmse, mae, mape, r2]
    })
    st.dataframe(metrics_df.style.format({'Value': '{:.4f}'}))
    
    # Residual analysis
    residuals = closing_prices[seq_length:] - y_pred_all_inv
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_res = plt.figure(figsize=(10,4))
        plt.plot(residuals, label='Residuals', color='purple')
        plt.axhline(y=0, color='gray', linestyle='--')
        plt.title(f'{company} Residual Plot: Actual - Predicted')
        plt.xlabel('Time')
        plt.ylabel('Residual Value')
        plt.legend()
        st.pyplot(fig_res)
    
    with col2:
        fig_hist = plt.figure(figsize=(10,4))
        plt.hist(residuals, bins=50, edgecolor='black', color='orchid')
        plt.title(f'{company} Histogram of Residuals')
        plt.xlabel('Residual Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        st.pyplot(fig_hist)
    
    # Future predictions
    last_50_days = closing_prices_scaled[-seq_length:]
    last_50_days = last_50_days.reshape(1, seq_length, 1)
    
    future_predictions = []
    for i in range(30):  
        next_day_price = model.predict(last_50_days)
        future_predictions.append(next_day_price[0, 0])
        last_50_days = np.append(last_50_days[:,1:,:], next_day_price.reshape(1,1,1), axis=1)
    
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1,1))
    
    fig_future = plt.figure(figsize=(10,4))
    plt.plot(range(1,31), future_predictions, marker='o', linestyle='dashed', color='red', label="Predicted Prices")
    plt.title(f"{company} Stock Prediction for Next 30 Days")
    plt.xlabel("Days Ahead")
    plt.ylabel("Stock Price")
    plt.legend()
    st.pyplot(fig_future)
    return {'predictions': y_pred_all,
        'residuals': residuals,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R²': r2,
        'train_loss': history.history['loss'][-1],
        'val_loss': history.history['val_loss'][-1]
        }



# Add this function to your existing dashboard code
def run_news_sentiment_analysis(company):
    st.subheader("News Sentiment Analysis")
    
    ticker = 'AAPL' if company == "Apple" else 'GOOGL'
    
    with st.spinner(f'Fetching news sentiment data for {ticker}...'):
        try:
            # Step 1: Scrape News Headlines from Finviz
            finviz_url = 'https://finviz.com/quote.ashx?t='
            url = finviz_url + ticker
            req = Request(url=url, headers={'user-agent': 'my-app'})
            response = urlopen(req)
            html = BeautifulSoup(response, 'html.parser')
            news_table = html.find(id='news-table')

            # Step 2: Parse News Data
            parse_data = []
            rows = news_table.findAll('tr')
            last_date = None
            for row in rows:
                title = row.a.text
                date_data = row.td.text.strip().split(' ')
                if len(date_data) == 1:
                    time = date_data[0]
                    date = last_date
                else:
                    date, time = date_data
                    last_date = date
                parse_data.append([ticker, date, time, title])

            # Step 3: Create DataFrame
            df = pd.DataFrame(parse_data, columns=['ticker', 'date', 'time', 'title'])

            # Fix "Today" and convert to datetime
            today_str = datetime.today().strftime('%b-%d-%y')
            df['date'] = df['date'].replace('Today', today_str)
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
            df['date'] = df['datetime'].dt.date  # Keep only date part for grouping

            # Step 4: Apply VADER Sentiment Analysis
            vader = SentimentIntensityAnalyzer()

            def get_sentiment_category(text):
                score = vader.polarity_scores(text)['compound']
                if score >= 0.05:
                    return 'positive'
                elif score <= -0.05:
                    return 'negative'
                else:
                    return 'neutral'

            df['sentiment'] = df['title'].apply(get_sentiment_category)

            # Step 5: Group and Count Sentiments by Date
            sentiment_counts = df.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
            
            # Also compute average compound score per date per sentiment
            df['compound'] = df['title'].apply(lambda x: vader.polarity_scores(x)['compound'])
            avg_scores = df.groupby(['date', 'sentiment'])['compound'].mean().unstack(fill_value=0)

            # Ensure we have all sentiment categories
            sentiment_order = ['negative', 'neutral', 'positive']
            for col in sentiment_order:
                if col not in sentiment_counts.columns:
                    sentiment_counts[col] = 0
                if col not in avg_scores.columns:
                    avg_scores[col] = 0

            # Plot Sentiment Trends
            fig, ax = plt.subplots(figsize=(10, 4))
            
            # Plot stacked bar chart
            sentiment_counts[sentiment_order].plot(kind='bar', stacked=True, ax=ax, 
                                                  color=['red', 'gray', 'green'])
            
            # Customize the plot
            ax.set_title(f'{company} News Sentiment Counts by Day')
            ax.set_ylabel('Number of Headlines')
            ax.set_xlabel('Date')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Add labels: count (avg_score)
            for idx, date in enumerate(sentiment_counts.index):
                y_offset = 0
                for sentiment in sentiment_order:
                    count = sentiment_counts.loc[date][sentiment]
                    if count > 0:
                        avg_score = avg_scores.loc[date][sentiment]
                        label = f"{count} ({avg_score:.2f})"
                        ax.text(idx, y_offset + count / 2, label, 
                                ha='center', va='center', fontsize=8, color='white')
                        y_offset += count
            
            st.pyplot(fig)

            # Display sentiment summary
            st.subheader("Sentiment Summary")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Sentiment Counts**")
                st.dataframe(sentiment_counts[sentiment_order].style.background_gradient(cmap='coolwarm'))
            
            with col2:
                st.write("**Average Compound Scores**")
                st.dataframe(avg_scores[sentiment_order].style.background_gradient(cmap='coolwarm'))
            
            # Display sample headlines
            st.subheader("Sample Headlines")
            sample_size = min(10, len(df))
            st.table(df[['date', 'title', 'sentiment']].head(sample_size))
            
        except Exception as e:
            st.error(f"Error fetching news sentiment data: {str(e)}")
            st.warning("Please check your internet connection and try again.")


# =============================================
# NEW ADVANCED ANALYSIS FUNCTION
# =============================================

def run_advanced_behavioral_analysis(ticker, company_name):
    """Complete analysis with all parameters"""
    st.subheader("📊 Advanced Behavioral Analysis")
    
    # Create columns for better layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📈 Key Metrics")
        
        # Get all metrics with fallbacks
        pe_ratio = get_pe_ratio(ticker)
        roe = get_roe(ticker)
        vix = yf.Ticker("^VIX").history(period="1d")['Close'].iloc[-1]
        iv = get_implied_volatility(ticker)
        sentiment = analyze_news_sentiment(ticker)
        
        # Display metrics in a clean table
        metrics_data = [
            {"Metric": "P/E Ratio", "Value": f"{pe_ratio:.2f}" if pe_ratio else "N/A", 
             "Benchmark": "25.00", "Status": "✅ Good" if pe_ratio and pe_ratio < 25 else "⚠️ High"},
            {"Metric": "ROE (%)", "Value": f"{roe:.2f}" if roe else "N/A", 
             "Benchmark": "20.00", "Status": "✅ Strong" if roe and roe > 20 else "⚠️ Weak"},
            {"Metric": "VIX", "Value": f"{vix:.2f}", 
             "Benchmark": "20.00", "Status": "✅ Low" if vix < 20 else "⚠️ High"},
            {"Metric": "Implied Vol (%)", "Value": f"{iv*100:.2f}" if iv else "N/A", 
             "Benchmark": "25.00", "Status": "✅ Low" if iv and iv*100 < 25 else "⚠️ High"},
            {"Metric": "Sentiment Score", "Value": f"{sentiment['compound_mean']:.2f}" if sentiment else "N/A", 
             "Benchmark": "0.10", "Status": "✅ Positive" if sentiment and sentiment['compound_mean'] > 0.1 else "⚠️ Neutral/Negative"}
        ]
        
        st.dataframe(
            pd.DataFrame(metrics_data).style.apply(
                lambda x: ['background-color: #e6ffe6' if '✅' in v else 
                          'background-color: #fff2cc' for v in x],
                subset=["Status"]
            ),
            hide_index=True
        )
    
    with col2:
        st.markdown("### 🚦 Investment Signal")
        
        # Calculate composite score (0-100)
        score = 0
        max_score = 0
        
        if pe_ratio:
            score += 20 if pe_ratio < 25 else 0
            max_score += 20
            
        if roe:
            score += 20 if roe > 20 else 0
            max_score += 20
            
        score += 20 if vix < 20 else 0
        max_score += 20
        
        if iv:
            score += 20 if iv*100 < 25 else 0
            max_score += 20
            
        if sentiment:
            score += 20 if sentiment['compound_mean'] > 0.1 else 0
            max_score += 20
        
        # Normalize score
        if max_score > 0:
            final_score = (score / max_score) * 100
        else:
            final_score = 0
        
        # Display gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=final_score,
            title="Bullish Score",
            gauge={
                'axis': {'range': [0, 100]},
                'steps': [
                    {'range': [0, 40], 'color': "red"},
                    {'range': [40, 70], 'color': "orange"},
                    {'range': [70, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.8,
                    'value': final_score
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendation
        if final_score >= 70:
            st.success("### 🚀 STRONG BUY")
            st.markdown("""
            **Rationale:**  
            Multiple positive indicators align suggesting strong upside potential.
            """)
        elif final_score >= 40:
            st.warning("### 🔄 HOLD")
            st.markdown("""
            **Rationale:**  
            Mixed signals - maintain current position until clearer trend emerges.
            """)
        else:
            st.error("### ⚠️ SELL")
            st.markdown("""
            **Rationale:**  
            Multiple risk factors detected - consider reducing exposure.
            """)
    
    # Behavioral insights
    st.markdown("---")
    st.subheader("🧠 Behavioral Insights")
    
    if pe_ratio and pe_ratio > 25:
        st.markdown("- **High P/E** suggests investors may be overpaying due to FOMO (Fear Of Missing Out)")
    if roe and roe < 20:
        st.markdown("- **Low ROE** indicates the company might not be using investments efficiently")
    if vix > 20:
        st.markdown("- **High VIX** shows increased market fear, which can lead to emotional decisions")
    if iv and iv*100 > 25:
        st.markdown("- **High Implied Volatility** suggests traders expect big price swings")
    if sentiment and sentiment['compound_mean'] <= 0.1:
        st.markdown("- **Neutral/Negative Sentiment** may indicate lack of positive catalysts")

# ===========================================
# END OF NEW FUNCTION
# ===========================================

def run_sp500_comparison_analysis(df,company):
    """Run complete S&P 500 comparison analysis for the dashboard"""
    st.subheader("📊 S&P 500 Comparative Analysis (5 Years)")
    
    ticker = 'AAPL' if company == "Apple" else 'GOOGL'
    
    with st.spinner(f'Fetching 5 years of historical data for {company}...'):
        try:
            # Configuration
            end_date = datetime.today()
            start_date = end_date - timedelta(days=5*365)  # 5 years of data

            # Fetch data
            data = {}
            data['sp500'] = yf.download("^GSPC", start=start_date, end=end_date)
            data[ticker] = yf.download(ticker, start=start_date, end=end_date)
            
            # Calculate daily returns
            data[ticker]['Daily_Return'] = data[ticker]['Close'].pct_change()
            data['sp500']['Daily_Return'] = data['sp500']['Close'].pct_change()

            # Calculate metrics
            stock = data[ticker]
            sp500 = data['sp500']
            
            # 1. Annualized Returns
            stock_returns = stock['Daily_Return'].dropna()
            sp500_returns = sp500['Daily_Return'].dropna()
            stock_annual_return = (1 + stock_returns.mean())**252 - 1
            sp500_annual_return = (1 + sp500_returns.mean())**252 - 1
            
            # 2. Risk-Adjusted Returns (Sharpe Ratio)
            risk_free_rate = 0.02
            stock_sharpe = (stock_annual_return - risk_free_rate) / (stock_returns.std() * np.sqrt(252))
            sp500_sharpe = (sp500_annual_return - risk_free_rate) / (sp500_returns.std() * np.sqrt(252))
            
            # 3. Maximum Drawdown
            cumulative = (1 + stock_returns).cumprod()
            peak = cumulative.cummax()
            drawdown = (cumulative - peak) / peak
            max_drawdown = drawdown.min()
            
            # 4. Current Technicals - RSI
            close_prices = stock['Close'].values.flatten()
            rsi_values = RSI(close_prices, timeperiod=14)
            current_rsi = rsi_values[-1] if len(rsi_values) > 0 else 50.0
            
            # 5. Relative Strength
            if len(stock['Close']) >= 252 and len(sp500['Close']) >= 252:
                stock_ratio = float(stock['Close'].iloc[-1]) / float(stock['Close'].iloc[-252])
                sp500_ratio = float(sp500['Close'].iloc[-1]) / float(sp500['Close'].iloc[-252])
                relative_strength = stock_ratio / sp500_ratio
            else:
                relative_strength = 1.0
            
            # Generate investment decision
            conditions = [
                (relative_strength > 1.2) and (current_rsi < 60),
                (relative_strength < 0.8) and (current_rsi > 70),
                (stock_annual_return > sp500_annual_return + 0.05) and (stock_sharpe > 1),
                (max_drawdown < -0.30) and (current_rsi > 65),
                (stock_sharpe < 0.5) and (stock_annual_return < sp500_annual_return)
            ]
            
            decisions = [
                "STRONG BUY: Outperforming market with reasonable valuation",
                "SELL: Underperforming and overbought",
                "BUY: Consistent outperformer with good risk-adjusted returns",
                "CAUTION: High historical volatility and currently overbought",
                "AVOID: Poor risk-adjusted returns compared to market"
            ]
            
            decision = "HOLD: Neutral position relative to market"
            for condition, dec in zip(conditions, decisions):
                if condition:
                    decision = dec
                    break
            
            # Display metrics in columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("5-Year Annualized Return", 
                         f"{stock_annual_return*100:.1f}%", 
                         delta=f"{stock_annual_return*100 - sp500_annual_return*100:.1f}% vs S&P 500")
                
                st.metric("Risk-Adjusted Return (Sharpe)", 
                         f"{stock_sharpe:.2f}", 
                         delta=f"{stock_sharpe - sp500_sharpe:.2f} vs S&P 500")
                
                st.metric("Maximum Drawdown", f"{max_drawdown*100:.1f}%")
            
            with col2:
                st.metric("Current RSI (14-day)", f"{current_rsi:.1f}",
                         delta="Overbought (>70)" if current_rsi > 70 else 
                               "Oversold (<30)" if current_rsi < 30 else "Neutral")
                
                st.metric("1-Year Relative Strength", f"{relative_strength:.2f}",
                         delta="Outperforming (>1)" if relative_strength > 1 else "Underperforming")
                
                st.metric("Investment Decision", decision, delta="Recommendation")
            
            # Plot comparison
#            st.subheader(f"5-Year Performance: {company} vs S&P 500")
            
            # Normalize to percentage scale
#           norm_stock = (stock['Close'] / stock['Close'].iloc[0]) * 100
#           norm_sp500 = (sp500['Close'] / sp500['Close'].iloc[0]) * 100
            
#            fig = go.Figure()
            
            # Add stock trace
#            fig.add_trace(go.Scatter(
#                x=norm_stock.index,
#                y=norm_stock,
#                name=f"{company}",
#               line=dict(color='#1f77b4', width=2),
#                hovertemplate='%{x|%b %d, %Y}<br>%{y:.1f}%'
#            ))
            
            # Add S&P 500 trace
#            fig.add_trace(go.Scatter(
#                x=norm_sp500.index,
#                y=norm_sp500,
#                name="S&P 500",
#                line=dict(color='#2ca02c', width=2, dash='dash'),
#                hovertemplate='%{x|%b %d, %Y}<br>%{y:.1f}%'
#            ))
            
#            fig.update_layout(
#                height=500,
#                xaxis_title="Date",
#                yaxis_title="Normalized Performance (100 = Starting Value)",
#                legend=dict(
#                    orientation="h",
#                    yanchor="bottom",
#                    y=1.02,
#                    xanchor="right",
#                    x=1
#                ),
            #     hovermode="x unified",
            #     margin=dict(l=20, r=20, t=40, b=20),
            #     plot_bgcolor='rgba(240,240,240,0.8)'
            # )
            
            # # Add range slider
            # fig.update_xaxes(
            #     rangeslider_visible=True,
            #     rangeselector=dict(
            #         buttons=list([
            #             dict(count=1, label="1Y", step="year", stepmode="backward"),
            #             dict(count=3, label="3Y", step="year", stepmode="backward"),
            #             dict(step="all", label="5Y")
            #         ])
            #     )
            # )
            
            # st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error in S&P 500 comparison analysis: {str(e)}")



# def run_sp500_comparison_analysis(df, company):
#     """Display S&P 500 comparison with custom graph and metrics"""
#     st.subheader("📊 S&P 500 Comparative Analysis (5 Years)")
    
#     # --- Part 1: Display Your Custom Graph ---
#     try:
#         if company == "Apple":
#             img_path = "C:/Users/Dell/Dissertation_market_value/Apple_image/Apple_sp500.png"
#         else:
#             img_path = "C:/Users/Dell/Dissertation_market_value/Google_image/Google_sp500.png"
        
#         st.image(img_path, 
#                 caption=f"{company} vs S&P 500 (2019-2024)",
#                 use_column_width=True)
#     except Exception as e:
#         st.warning(f"Could not display graph: {str(e)}")
    
#     # --- Part 2: Calculate and Display Metrics ---
#     ticker = 'AAPL' if company == "Apple" else 'GOOGL'
#     end_date = datetime(2024, 12, 31)
#     start_date = end_date - timedelta(days=5*365)  # Exactly 5 years
    
#     try:
#         # Fetch data safely
#         stock = yf.download(ticker, start=start_date, end=end_date, progress=False)
#         sp500 = yf.download("^GSPC", start=start_date, end=end_date, progress=False)
        
#         # Verify we got data
#         if stock.empty or sp500.empty:
#             st.error("No data available for the selected period")
#             return
        
#         # Get scalar values for calculations
#         stock_first = float(stock['Close'].iloc[0])
#         stock_last = float(stock['Close'].iloc[-1])
#         sp500_first = float(sp500['Close'].iloc[0])
#         sp500_last = float(sp500['Close'].iloc[-1])
        
#         # Calculate returns
#         stock_return = (stock_last / stock_first) - 1
#         sp500_return = (sp500_last / sp500_first) - 1
        
#         # Annualize returns
#         stock_annualized = ((1 + stock_return) ** (1/5)) - 1
#         sp500_annualized = ((1 + sp500_return) ** (1/5)) - 1
        
#         # Calculate relative strength (1Y)
#         if len(stock) >= 252 and len(sp500) >= 252:
#             stock_1y_first = float(stock['Close'].iloc[-252])
#             sp500_1y_first = float(sp500['Close'].iloc[-252])
#             rel_strength = (stock_last/stock_1y_first) / (sp500_last/sp500_1y_first)
#         else:
#             rel_strength = 1.0
        
#         # --- Display Metrics ---
#         st.markdown("**Performance Metrics**")
        
#         col1, col2 = st.columns(2)
#         with col1:
#             st.metric(f"{company} Total Return", 
#                      f"{stock_return*100:.1f}%",
#                      delta=f"Annualized: {stock_annualized*100:.1f}%/yr")
            
#         with col2:
#             st.metric("S&P 500 Total Return",
#                      f"{sp500_return*100:.1f}%",
#                      delta=f"Annualized: {sp500_annualized*100:.1f}%/yr")
        
#         st.metric("Relative Strength (1Y)",
#                  f"{rel_strength:.2f}",
#                  "Outperforming" if rel_strength > 1 else "Underperforming")
        
#     except Exception as e:
#         st.error(f"Error calculating metrics: {str(e)}")
# Function to run behavioral analysis
def run_behavioral_analysis(df, company):
    st.header("📊 Behavioral Analysis")
    
    # VIX and PE Analysis
    if 'vix_close' in df.columns and 'PE' in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            fig_vix = plt.figure(figsize=(10,4))
            ax1 = plt.gca()
            ax1.plot(df['DATE'], df['Close'], color='blue', label='Stock Price')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Stock Price', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            
            ax2 = ax1.twinx()
            ax2.plot(df['DATE'], df['vix_close'], color='red', label='VIX Close')
            ax2.set_ylabel('VIX Close', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            
            plt.title(f'{company} Stock Price and VIX Close Over Time')
            plt.grid(True)
            st.pyplot(fig_vix)
        
        with col2:
            fig_pe = plt.figure(figsize=(6,4))
            ax1 = plt.gca()
            ax1.plot(df['DATE'], df['Close'], color='blue', label='Stock Price')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Stock Price', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            
            ax2 = ax1.twinx()
            ax2.plot(df['DATE'], df['PE'], color='green', label='P/E Ratio')
            ax2.set_ylabel('P/E Ratio', color='green')
            ax2.tick_params(axis='y', labelcolor='green')
            
            plt.title(f'{company} Stock Price and P/E Ratio Over Time')
            plt.grid(True)
            st.pyplot(fig_pe)
    
    # ROE Analysis
    roe_col = 'ROE_Quarterly' if company == "Apple" else 'Quarterly_ROE (%)'
    date_col = 'Date_quaterly' if company == "Apple" else 'Date_net'
    
    if roe_col in df.columns and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col, roe_col, 'Close'])
        
        fig_roe = plt.figure(figsize=(10,4))
        ax1 = plt.gca()
        ax1.set_xlabel("Date")
        ax1.set_ylabel("ROE (%)", color='orange')
        ax1.plot(df[date_col], df[roe_col], marker='o', linestyle='-', color='orange', label='Quarterly ROE')
        ax1.tick_params(axis='y', labelcolor='orange')
        
        ax2 = ax1.twinx()
        ax2.set_ylabel("Stock Price (Close)", color='blue')
        ax2.plot(df[date_col], df['Close'], marker='s', linestyle='--', color='blue', label='Stock Price (Close)')
        ax2.tick_params(axis='y', labelcolor='blue')
        
        plt.title(f"Quarterly ROE and {company} Stock Price Over Time")
        plt.grid(True)
        st.pyplot(fig_roe)
        
        # Yearly ROE
        df['Year'] = df[date_col].dt.year
        yearly_roe = df.groupby('Year')[roe_col].mean().reset_index()
        
        fig_yearly = plt.figure(figsize=(10,4))
        plt.bar(yearly_roe['Year'].astype(str), yearly_roe[roe_col], color='skyblue')
        plt.title(f"Average Yearly ROE ({company})")
        plt.xlabel("Year")
        plt.ylabel("Average ROE (%)")
        plt.grid(axis='y')
        st.pyplot(fig_yearly)

    # implied volality 
        # Volatility Analysis Section
       # Volatility Analysis Section
    st.subheader("Implied Volatility Analysis")
    
    try:
        ticker = 'AAPL' if company == "Apple" else 'GOOGL'
        
        with st.spinner(f'Fetching current {company} volatility data...'):
            # Get current market data
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1d")
            
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                implied_volatility = get_implied_volatility(ticker)
                
                if implied_volatility is not None:
                    fluctuation, lower, upper = calculate_expected_range(current_price, implied_volatility)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Current Price", f"${current_price:.2f}")
                        st.metric("30-Day Implied Volatility", f"{implied_volatility:.2%}")
                    
                    with col2:
                        st.metric("Expected 30-Day Fluctuation", f"±${fluctuation:.2f}")
                        st.metric("Expected Price Range", f"${lower:.2f} - ${upper:.2f}")
                    
                    # Plot the volatility analysis
                    st.write("### Expected Price Movement")
                    fig, upper, lower, prices = plot_volatility_analysis(current_price, fluctuation, company)
                    st.pyplot(fig)
                else:
                    st.warning("Could not fetch implied volatility data. The website might be temporarily unavailable.")
            else:
                st.warning("Could not fetch current price data. Please check your internet connection.")
                
    except Exception as e:
        st.error(f"Error fetching volatility data: {str(e)}")
        st.warning("Please ensure you have an active internet connection to fetch the latest market data.")
    
    run_sp500_comparison_analysis(df, company)
    st.markdown("---") 
    run_news_sentiment_analysis(company) 
    st.markdown("---")
    ticker = 'AAPL' if company == "Apple" else 'GOOGL'
    run_advanced_behavioral_analysis(ticker, company)



# Company Selection
#company_choice = st.selectbox("Select Company for Analysis:", ["Apple", "Google"])

# File Upload - Main Dataset
#uploaded_file = st.file_uploader(f"Upload {company_choice} stock data (CSV format)", type="csv")
# ... (keep all your existing imports and helper functions) ...

# Modify the main content section where you create the tabs
if uploaded_file:
    try:
        # Read and preprocess main dataset
        df = pd.read_csv(uploaded_file)
        df['DATE'] = pd.to_datetime(df['DATE'])
        df.set_index('DATE', inplace=True)

        df['Close'] = df['Close'].ffill().bfill()
        df = df.dropna(subset=['Close'])

        if len(df) < 60:
            st.error("Dataset must contain at least 60 values.")
        else:
            # Create tabs - now with 5 tabs including Validation
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "ARIMA Model", 
                "LSTM Model", 
                "Model Comparison", 
                "Behavioral Analysis",
                "Validation Analysis"  # New tab for validation
            ])

            with tab1:
                st.header(f"{company_choice} - ARIMA Model Analysis")
                arima_results = run_arima_analysis(df, company_choice)

            with tab2:
                st.header(f"{company_choice} - LSTM Model Analysis")
                lstm_results = run_lstm_analysis(df, company_choice)

            with tab3:
                st.header(f"{company_choice} - Model Comparison")
                if 'arima_results' not in locals():
                    arima_results = run_arima_analysis(df, company_choice)
                if 'lstm_results' not in locals():
                    lstm_results = run_lstm_analysis(df, company_choice)
                show_model_comparison(arima_results, lstm_results, df, company_choice)

            with tab4:
                st.header(f"{company_choice} - Behavioral Analysis")
                run_behavioral_analysis(df.reset_index(), company_choice)
                
            with tab5:
               st.header("Validation Analysis")
               validation_file = st.file_uploader(
                           "Upload Validation Dataset (CSV format)", 
                            type=["csv"], 
                             key="validation"
               )
    
    

               if validation_file:
                   try:
            # Read and preprocess validation dataset
                     df_val = pd.read_csv(validation_file)
                     df_val['DATE'] = pd.to_datetime(df_val['DATE'])
                     df_val.set_index('DATE', inplace=True)
                     df_val['Close'] = df_val['Close'].ffill().bfill()
                     df_val = df_val.dropna(subset=['Close'])

                     if len(df_val) < 60:
                         st.error("Validation dataset must contain at least 60 values.")
                     else:
                # =============================================
                # NEW: Detect company name from filename or let user select
                # =============================================
                # Option 1: Extract from filename (e.g., "META_stock.csv" → "META")
                          val_company = validation_file.name.split("_")[0].upper()
                
                # Option 2: Let user manually select (safer)
                # val_company = st.selectbox(
                #     "Select company for validation:",
                #     ["META", "AAPL", "GOOGL", "Other"],
                #     key="val_company"
                # )
                
                          st.markdown(f"## 🔍 Validating on {val_company} Dataset")

                # Run ARIMA on Validation Dataset
                          st.subheader("ARIMA Validation Results")
                          arima_val_results = run_arima_analysis(df_val, val_company)  # Pass dynamic name

                # Run LSTM on Validation Dataset
                          st.subheader("LSTM Validation Results")
                          lstm_val_results = run_lstm_analysis(df_val, val_company)  # Pass dynamic name

                # Compare Metrics Old vs New
                          st.markdown("## 📊 Model Performance Comparison (Original vs Validation)")

                          comparison_data ={
                              "Metric": ["RMSE", "MAE", "MAPE (%)", "R² Score"],
                                    "ARIMA (Original)": [
                                    f"{arima_results['RMSE']:.4f}", 
                                    f"{arima_results['MAE']:.4f}", 
                                    f"{arima_results['MAPE']:.2f}", 
                                    f"{arima_results['R²']:.4f}"
                                ],
                                "ARIMA (Validation)": [
                                    f"{arima_val_results['RMSE']:.4f}", 
                                    f"{arima_val_results['MAE']:.4f}", 
                                    f"{arima_val_results['MAPE']:.2f}", 
                                    f"{arima_val_results['R²']:.4f}"
                                ],
                                "LSTM (Original)": [
                                    f"{lstm_results['RMSE']:.4f}", 
                                    f"{lstm_results['MAE']:.4f}", 
                                    f"{lstm_results['MAPE']:.2f}", 
                                    f"{lstm_results['R²']:.4f}"
                                ],
                                "LSTM (Validation)": [
                                    f"{lstm_val_results['RMSE']:.4f}", 
                                    f"{lstm_val_results['MAE']:.4f}", 
                                    f"{lstm_val_results['MAPE']:.2f}", 
                                    f"{lstm_val_results['R²']:.4f}"
                                ]
                              
                          }
                                    
                          

                          comparison_df = pd.DataFrame(comparison_data)
                          st.dataframe(comparison_df.style.format(precision=4), hide_index=True)

                # Visualize Actual vs Predicted Side by Side
                          st.markdown("## 📈 Actual vs Predicted - Side by Side")

                          col1, col2 = st.columns(2)
                          with col1:
                              st.write("### ARIMA Model (Validation Dataset)")
                              fig_arima_val = plt.figure(figsize=(8, 4))
                              plt.plot(df_val['Close'], label='Actual Price', color='blue')
                              plt.plot(df_val.index[1:len(arima_val_results['predictions'])+1], 
                                       arima_val_results['predictions'], 
                                       label='ARIMA Prediction', color='red', linestyle='dotted')
                              plt.title(f"{val_company} ARIMA: Actual vs Predicted (Validation)")  # Dynamic title
                              plt.xlabel("Date")
                              plt.ylabel("Price")
                              plt.legend()
                              plt.grid(True)
                              st.pyplot(fig_arima_val)

                          with col2:
                              st.write("### LSTM Model (Validation Dataset)")
                              fig_lstm_val = plt.figure(figsize=(8, 4))
                              plt.plot(df_val['Close'].values, label='Actual Price', color='blue')
                              aligned_preds_val = np.empty_like(df_val['Close'].values)
                              aligned_preds_val[:] = np.nan
                              aligned_preds_val[60:] = lstm_val_results['predictions'].flatten()
                              plt.plot(aligned_preds_val, label='LSTM Prediction', 
                                         color='green', linestyle='dashed')
                              plt.title(f"{val_company} LSTM: Actual vs Predicted (Validation)")  # Dynamic title
                              plt.xlabel("Time Step")
                              plt.ylabel("Price")
                              plt.legend()
                              plt.grid(True)
                              st.pyplot(fig_lstm_val)

                   except Exception as e:
                       st.error(f"Error processing validation file: {str(e)}")
               else:
                   st.info("Upload a validation dataset to compare model performance.")

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
else:
    st.info(f"Please upload {company_choice} stock data file to proceed with the analysis.")

# # Main content based on selection
# # Main content based on selection
# if analysis_type == "Apple Stock Analysis":
#     st.header("🍎 Apple Stock Analysis")
    
#     # File uploader
#     uploaded_file = st.file_uploader("Upload Apple stock data (AAPL_stocks.csv)", type="csv")
    
#     if uploaded_file is not None:
#         try:
#             # Read the uploaded file
#             df = pd.read_csv(uploaded_file)
            
#             # Convert DATE column to datetime and set as index
#             df['DATE'] = pd.to_datetime(df['DATE'])
#             df.set_index('DATE', inplace=True)
            
#             # Model selection - Updated to include comparison option
#             model_choice = st.radio("Select Analysis Mode:", 
#                                    ["Single Model (LSTM)", "Single Model (ARIMA)", "Compare Both Models"])
            
#             if model_choice == "Single Model (LSTM)":
#                 run_lstm_analysis(df, "Apple")
#             elif model_choice == "Single Model (ARIMA)":
#                 run_arima_analysis(df, "Apple")
#             elif model_choice == "Compare Both Models":
#                 with st.spinner("Running both models for Apple. This may take a few minutes..."):
#                     col1, col2 = st.columns(2)
#                     with col1:
#                         arima_results = run_arima_analysis(df, "Apple")
#                     with col2:
#                         lstm_results = run_lstm_analysis(df, "Apple")
                
#                 # Show comparison window
#                 show_model_comparison(arima_results, lstm_results, df, "Apple")
            
#             # Behavioral analysis checkbox
#             if st.checkbox("Show Behavioral Analysis"):
#                 run_behavioral_analysis(df.reset_index(), "Apple")
                
#         except Exception as e:
#             st.error(f"Error processing file: {str(e)}")
#     else:
#         st.info("Please upload Apple stock data file to proceed with the analysis.")

# elif analysis_type == "Google Stock Analysis":
#     st.header("🔍 Google Stock Analysis")
    
#     # File uploader
#     uploaded_file = st.file_uploader("Upload Google stock data (GOOGLE_stocks.csv)", type="csv")
    
#     if uploaded_file is not None:
#         try:
#             # Read the uploaded file
#             df = pd.read_csv(uploaded_file)
            
#             # Convert DATE column to datetime and set as index
#             df['DATE'] = pd.to_datetime(df['DATE'])
#             df.set_index('DATE', inplace=True)

#             # Handling missing data
#             df['Close'] = df['Close'].ffill().bfill()
#             df = df.dropna(subset=['Close'])

#             if len(df) < 60:
#                 st.error("Not enough data points. Dataset must contain at least 60 values.")
#             else:
#                 # Model selection - Updated to include comparison option
#                 model_choice = st.radio("Select Analysis Mode:", 
#                                        ["LSTM Model", "ARIMA Model"])
                
#                 if model_choice == "Single Model (LSTM)":
#                     run_lstm_analysis(df, "Google")
#                 elif model_choice == "Single Model (ARIMA)":
#                     run_arima_analysis(df, "Google")
#                 elif model_choice == "Compare Both Models":
#                     with st.spinner("Running both models for Google. This may take a few minutes..."):
#                         col1, col2 = st.columns(2)
#                         with col1:
#                             arima_results = run_arima_analysis(df, "Google")
#                         with col2:
#                             lstm_results = run_lstm_analysis(df, "Google")
                    
#                     # Show comparison window
#                     show_model_comparison(arima_results, lstm_results, df, "Google")
                
#                 # Behavioral analysis checkbox
#                 if st.checkbox("Show Behavioral Analysis"):
#                     run_behavioral_analysis(df.reset_index(), "Google")
                
#         except Exception as e:
#             st.error(f"Error processing file: {str(e)}")
#     else:
#         st.info("Please upload Google stock data file to proceed with the analysis.")
# """
# Footer
st.markdown("---")
st.markdown("""
#**Final Project** - Stock Market Analysis using ARIMA and LSTM Models  
#Developed for Dissertation  
#*University of Leicester*  
""")
