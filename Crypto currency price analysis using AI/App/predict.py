import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd

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

# Keep the last 29 days of historical data
historical_data = df[numeric_cols].iloc[-29:].values  

# Get User Input (Single Day)
user_input = [
    "388.10","405.48","387.51","402.97","402.97","54,824,800"
]  

# Convert user input to float
user_input = [float(val.replace(",", "")) for val in user_input]

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

# Output Prediction
print(f"Predicted Bitcoin Price: ${predicted_price:.2f}")
