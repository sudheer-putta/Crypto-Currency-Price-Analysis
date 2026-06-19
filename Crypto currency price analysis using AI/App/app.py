from flask import Flask, render_template, request
import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import os
import glob
from datetime import datetime

import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

# Load the trained model
class GNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNNBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Reshape for GNN (batch, features, time)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.permute(0, 2, 1)  # Reshape back
        return x

# Hybrid Model: "GNN + LSTM + Transformer
class HybridModel(nn.Module):
    def __init__(self, input_dim, cnn_out, lstm_hidden, transformer_heads):
        super(HybridModel, self).__init__()
        
        # "GNN" Layer
        self.gnn = GNNBlock(input_dim, cnn_out)
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_size=cnn_out, hidden_size=lstm_hidden, batch_first=True)
        
        # Transformer Layer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=lstm_hidden, nhead=transformer_heads), num_layers=2)
        
        # Fully Connected Layer
        self.fc = nn.Linear(lstm_hidden, 1)

    def forward(self, x):
        x = self.gnn(x)
        x, _ = self.lstm(x)
        x = self.transformer(x)
        x = self.fc(x[:, -1, :])  # Get last output from LSTM sequence
        return x

# Load Model & Scaler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridModel(input_dim=6, cnn_out=32, lstm_hidden=64, transformer_heads=4)
model.load_state_dict(torch.load("bitcoin_gnn_lstm_transformer.pth", map_location=device))
model.to(device)
model.eval()

scaler = joblib.load("scaler.pkl")

# Load Historical Data (Last 29 Days)
df = pd.read_csv("bitcoin_history.csv")
df['Date'] = pd.to_datetime(df['Date'])
numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

for col in numeric_cols:
    df[col] = df[col].astype(str).str.replace(',', '').astype(float)

df = df.sort_values('Date')

@app.route("/", methods=["GET", "POST"])
@app.route("/home", methods=["GET", "POST"])
def index():
    predicted_price = None

    if request.method == "POST":
        # Get user input from the form
        user_input = [
            request.form["open_price"],
            request.form["high_price"],
            request.form["low_price"],
            request.form["close_price"],
            request.form["adj_close_price"],
            request.form["volume"]
        ]

        # Convert user input to float
        user_input = [float(val.replace(",", "")) for val in user_input]

        # Keep the last 29 days of historical data
        historical_data = df[numeric_cols].iloc[-29:].values  

        # Append user input to historical data
        full_input = np.vstack([historical_data, user_input])

        # Scale input using the same scaler from training
        full_input = scaler.transform(full_input)

        # Convert input to tensor
        input_tensor = torch.tensor(full_input, dtype=torch.float32).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            prediction = model(input_tensor)

        # Convert prediction back to original scale
        predicted_price = scaler.inverse_transform([[0, 0, 0, prediction.item(), 0, 0]])[0][3]

    return render_template("index.html", predicted_price=predicted_price)

# Create directory for plots
PLOT_DIR = "static/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Load the dataset
df = pd.read_csv("bitcoin_history.csv")

# Data Cleaning
df['Date'] = pd.to_datetime(df['Date'])
numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
for col in numeric_columns:
    df[col] = df[col].str.replace(',', '').astype(float)

# Set seaborn style
sns.set(style='darkgrid')

def generate_plots():
    plot_paths = []

    for file in glob.glob(os.path.join(PLOT_DIR, "*.png")):
        os.remove(file)

    # 1. Bitcoin Price Over Time
    plt.figure(figsize=(12,6))
    plt.plot(df['Date'], df['Close'], label='Closing Price', color='blue')
    plt.xlabel('Date')
    plt.ylabel('Closing Price (USD)')
    plt.title('Bitcoin Price Over Time')
    plt.legend()
    path = os.path.join(PLOT_DIR, 'bitcoin_price.png')
    plt.savefig(path)
    plot_paths.append(path)
    plt.close()

    # 2. Candlestick OHLC Plot
    plt.figure(figsize=(12,6))
    plt.plot(df['Date'], df['Open'], label='Open', linestyle='dashed', alpha=0.6)
    plt.plot(df['Date'], df['High'], label='High', linestyle='dashed', alpha=0.6)
    plt.plot(df['Date'], df['Low'], label='Low', linestyle='dashed', alpha=0.6)
    plt.plot(df['Date'], df['Close'], label='Close', color='black')
    plt.title('Bitcoin OHLC Prices Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    path = os.path.join(PLOT_DIR, 'ohlc.png')
    plt.savefig(path)
    plot_paths.append(path)
    plt.close()

    # 3. Moving Averages
    df['SMA_30'] = df['Close'].rolling(window=30).mean()
    df['SMA_100'] = df['Close'].rolling(window=100).mean()
    plt.figure(figsize=(12,6))
    plt.plot(df['Date'], df['Close'], label='Closing Price', color='blue', alpha=0.6)
    plt.plot(df['Date'], df['SMA_30'], label='30-day SMA', color='orange')
    plt.plot(df['Date'], df['SMA_100'], label='100-day SMA', color='red')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.title('Bitcoin Moving Averages')
    plt.legend()
    path = os.path.join(PLOT_DIR, 'moving_averages.png')
    plt.savefig(path)
    plot_paths.append(path)
    plt.close()

    # 4. Histogram of Closing Prices
    plt.figure(figsize=(10,5))
    sns.histplot(df['Close'], bins=50, kde=True, color='blue')
    plt.xlabel('Closing Price (USD)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Bitcoin Closing Prices')
    path = os.path.join(PLOT_DIR, 'histogram.png')
    plt.savefig(path)
    plot_paths.append(path)
    plt.close()

    # 5. Box Plot: Price Distribution
    plt.figure(figsize=(10,5))
    sns.boxplot(y=df['Close'], color='lightblue')
    plt.ylabel('Price (USD)')
    plt.title('Bitcoin Price Distribution (Box Plot)')
    path = os.path.join(PLOT_DIR, 'boxplot.png')
    plt.savefig(path)
    plot_paths.append(path)
    plt.close()

    # 6. Scatter Plot: Volume vs Closing Price
    plt.figure(figsize=(10,5))
    sns.scatterplot(x=df['Volume'], y=df['Close'], alpha=0.5, color='purple')
    plt.xlabel('Volume')
    plt.ylabel('Closing Price (USD)')
    plt.title('Bitcoin Trading Volume vs Closing Price')
    plt.xscale('log')
    path = os.path.join(PLOT_DIR, 'scatter_volume_price.png')
    plt.savefig(path)
    plot_paths.append(path)
    plt.close()

    # 7. Correlation Heatmap
    plt.figure(figsize=(8,6))
    corr = df[numeric_columns].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap of Bitcoin Data')
    path = os.path.join(PLOT_DIR, 'heatmap.png')
    plt.savefig(path)
    plot_paths.append(path)
    plt.close()

    # 8. Daily Returns Over Time
    df['Daily Return'] = df['Close'].pct_change()
    plt.figure(figsize=(12,6))
    sns.lineplot(x=df['Date'], y=df['Daily Return'], color='red')
    plt.xlabel('Date')
    plt.ylabel('Daily Return')
    plt.title('Bitcoin Daily Returns Over Time')
    path = os.path.join(PLOT_DIR, 'daily_returns.png')
    plt.savefig(path)
    plot_paths.append(path)
    plt.close()

    # 9. Rolling Volatility (30-day)
    df['Rolling Volatility'] = df['Daily Return'].rolling(window=30).std()
    plt.figure(figsize=(12,6))
    plt.plot(df['Date'], df['Rolling Volatility'], color='purple')
    plt.xlabel('Date')
    plt.ylabel('Rolling Volatility')
    plt.title('Bitcoin 30-day Rolling Volatility')
    path = os.path.join(PLOT_DIR, 'rolling_volatility.png')
    plt.savefig(path)
    plot_paths.append(path)
    plt.close()

    # 10. Trading Volume Moving Average
    df['Volume_SMA_30'] = df['Volume'].rolling(window=30).mean()
    plt.figure(figsize=(12,6))
    plt.plot(df['Date'], df['Volume'], label='Daily Volume', alpha=0.5)
    plt.plot(df['Date'], df['Volume_SMA_30'], label='30-day SMA Volume', color='red')
    plt.xlabel('Date')
    plt.ylabel('Trading Volume')
    plt.title('Bitcoin Trading Volume with 30-day SMA')
    plt.legend()
    path = os.path.join(PLOT_DIR, 'volume_moving_avg.png')
    plt.savefig(path)
    plot_paths.append(path)
    plt.close()

    # 11. Price Momentum
    df['Momentum'] = df['Close'] - df['Open']
    plt.figure(figsize=(12,6))
    plt.plot(df['Date'], df['Momentum'], label='Price Momentum', color='green')
    plt.xlabel('Date')
    plt.ylabel('Momentum (Close - Open)')
    plt.title('Bitcoin Price Momentum Over Time')
    plt.legend()
    path = os.path.join(PLOT_DIR, 'momentum.png')
    plt.savefig(path)
    plot_paths.append(path)
    plt.close()

    # 12. Relative Strength Index (RSI)
    window_length = 14
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window_length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window_length).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    plt.figure(figsize=(12,6))
    plt.plot(df['Date'], df['RSI'], label='RSI', color='orange')
    plt.axhline(70, linestyle='--', color='red', alpha=0.7)
    plt.axhline(30, linestyle='--', color='green', alpha=0.7)
    plt.xlabel('Date')
    plt.ylabel('RSI Value')
    plt.title('Bitcoin Relative Strength Index (RSI)')
    plt.legend()
    path = os.path.join(PLOT_DIR, 'rsi.png')
    plt.savefig(path)
    plot_paths.append(path)
    plt.close()

    return plot_paths

@app.route("/analysis")
def analysis():
    plots = generate_plots()
    return render_template("analysis.html", plot_paths=[p.replace("\\", "/") for p in plots], time=datetime.now())

if __name__ == "__main__":
    app.run(debug=True)