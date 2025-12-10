import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def create_sequences(data, seq_length):
    """Create time series sequences"""
    sequences = []
    targets = []
    
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        target = data[i+seq_length]
        sequences.append(seq)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)

def lstm_predict(csv_file, target_column, seq_length=10, epochs=50, test_size=0.2, learning_rate=0.001):
    """
    LSTM prediction function
    
    Parameters:
    csv_file: CSV file path
    target_column: Target column index (starting from 0)
    seq_length: Sequence length
    epochs: Number of training epochs
    test_size: Test set ratio
    learning_rate: Learning rate
    
    Returns:
    accuracy: Prediction accuracy
    """
    
    # 1. Read data
    try:
        df = pd.read_csv(csv_file, header=None)
    except:
        # If default reading fails, try different delimiters
        df = pd.read_csv(csv_file, header=None, sep=',')
    
    # 2. Extract target column
    if isinstance(target_column, int):
        if target_column >= len(df.columns):
            raise ValueError(f"Column index {target_column} out of range, data has only {len(df.columns)} columns")
        data = df.iloc[:, target_column].values.astype(float)
    else:
        raise TypeError("target_column should be an integer column index")
    
    # 3. Data normalization
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_normalized = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    
    # 4. Create sequences
    X, y = create_sequences(data_normalized, seq_length)
    
    # 5. Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    
    # 6. Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train).unsqueeze(-1)  # Add feature dimension
    y_train = torch.FloatTensor(y_train).unsqueeze(-1)
    X_test = torch.FloatTensor(X_test).unsqueeze(-1)
    y_test = torch.FloatTensor(y_test).unsqueeze(-1)
    
    # 7. Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel(input_size=1, hidden_size=50, num_layers=2, output_size=1).to(device)
    
    # 8. Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 9. Train model
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward propagation
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward propagation
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            test_losses.append(test_loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.6f}, Test Loss: {test_loss.item():.6f}')
    
    # 10. Prediction
    model.eval()
    with torch.no_grad():
        train_predict = model(X_train)
        test_predict = model(X_test)
    
    # Inverse normalization
    train_predict = scaler.inverse_transform(train_predict.cpu().numpy())
    y_train_actual = scaler.inverse_transform(y_train.cpu().numpy())
    test_predict = scaler.inverse_transform(test_predict.cpu().numpy())
    y_test_actual = scaler.inverse_transform(y_test.cpu().numpy())
    
    # 11. Calculate accuracy (using prediction direction accuracy)
    def direction_accuracy(actual, predicted):
        """Calculate accuracy of price change direction"""
        actual_changes = np.diff(actual.flatten())
        predicted_changes = np.diff(predicted.flatten())
        
        correct = ((actual_changes > 0) & (predicted_changes > 0)) | \
                  ((actual_changes < 0) & (predicted_changes < 0)) | \
                  ((actual_changes == 0) & (predicted_changes == 0))
        
        return np.mean(correct) * 100
    
    # Calculate direction accuracy for training and test sets
    train_accuracy = direction_accuracy(y_train_actual, train_predict)
    test_accuracy = direction_accuracy(y_test_actual, test_predict)
    
    print(f"\nTraining set direction accuracy: {train_accuracy:.2f}%")
    print(f"Test set direction accuracy: {test_accuracy:.2f}%")
    
    return test_accuracy

if __name__ == "__main__":
    accuracy = lstm_predict(
        csv_file=r'PingAnBank_historical_daily.csv',
        target_column=4,  # Predict the 3rd column (closing price)
        seq_length=3,     # Use 3 days of data to predict the next day
        epochs=100,       # Train for 100 epochs
        test_size=0.3,    # 30% as test set
        learning_rate=0.001
    )
    
    print(f"\nFinal test set accuracy: {accuracy:.2f}%")