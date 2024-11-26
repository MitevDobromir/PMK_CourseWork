import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

def load_and_prepare_data(file_path, max_rows=None):
    print("Loading data...")
    df = pd.read_csv(file_path)
    if max_rows:
        df = df[:max_rows]
    
    # Convert timestamp to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s', errors='coerce')
    df = df.dropna(subset=['Timestamp'])
    
    # Print original price range
    print(f"Original price range: {df['Close'].min():.2f} - {df['Close'].max():.2f}")
    
    # Create price scaler
    price_scaler = MinMaxScaler()
    scaled_close = price_scaler.fit_transform(df[['Close']])
    
    # Create feature scaler
    feature_scaler = MinMaxScaler()
    features = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
    scaled_features = feature_scaler.fit_transform(features)
    
    return scaled_features, price_scaler, df['Close'].values

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length, 3])  # Close price is at index 3
    return np.array(X), np.array(y)

def build_model(seq_length, n_features):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_length, n_features)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def plot_predictions(y_true, y_pred, train_size):
    plt.figure(figsize=(15, 6))
    
    # Convert to numpy arrays and flatten
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Print verification values
    print("\nFirst few actual values:", y_true[:5])
    print("First few predictions:", y_pred[:5])
    print(f"\nActual price range: {np.min(y_true):.2f} - {np.max(y_true):.2f}")
    print(f"Predicted price range: {np.min(y_pred):.2f} - {np.max(y_pred):.2f}")
    
    # Plot training data
    plt.plot(y_true[:train_size], color='blue', label='Training (Actual)', alpha=0.8)
    plt.plot(range(train_size), y_pred[:train_size], color='lightblue', 
             label='Training (Predicted)', alpha=0.8, linestyle='--')
    
    # Plot testing data
    plt.plot(range(train_size, len(y_true)), y_true[train_size:],
             color='green', label='Testing (Actual)', alpha=0.8)
    plt.plot(range(train_size, len(y_pred)), y_pred[train_size:],
             color='lightgreen', label='Testing (Predicted)', alpha=0.8, linestyle='--')
    
    plt.title('Bitcoin Price Predictions vs Actual Values')
    plt.xlabel('Time Steps')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Parameters
    FILE_PATH = 'C:/Users/User/Desktop/repos/PMK_Course_work/data/btcusd_1-min_data.csv'
    TRAIN_SPLIT = 0.8
    MAX_ROWS = 250000    # Much more data
    SEQUENCE_LENGTH = 60 # Longer patterns
    BATCH_SIZE = 128     # Larger batch for GPU efficiency
    EPOCHS = 30          # More training iterations
    
    try:
        # Load and prepare data
        data, price_scaler, original_prices = load_and_prepare_data(FILE_PATH, MAX_ROWS)
        print("Data shape:", data.shape)
        
        # Create sequences
        X, y = create_sequences(data, SEQUENCE_LENGTH)
        print("Sequence shape:", X.shape)
        print("Target shape:", y.shape)
        
        # Split into training and testing sets
        train_size = int(len(X) * TRAIN_SPLIT)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Build and train model
        print("\nBuilding and training model...")
        model = build_model(SEQUENCE_LENGTH, data.shape[1])
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            verbose=1
        )
        
        # Make predictions
        print("\nMaking predictions...")
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)
        
        # Prepare data for inverse scaling
        train_pred_full = np.zeros((len(train_predictions), data.shape[1]))
        test_pred_full = np.zeros((len(test_predictions), data.shape[1]))
        train_pred_full[:, 3] = train_predictions.flatten()  # Close price at index 3
        test_pred_full[:, 3] = test_predictions.flatten()
        
        # Inverse transform predictions
        train_predictions = price_scaler.inverse_transform(train_pred_full)[:, 3]
        test_predictions = price_scaler.inverse_transform(test_pred_full)[:, 3]
        
        # Prepare actual values
        y_train_full = np.zeros((len(y_train), data.shape[1]))
        y_test_full = np.zeros((len(y_test), data.shape[1]))
        y_train_full[:, 3] = y_train
        y_test_full[:, 3] = y_test
        
        # Inverse transform actual values
        y_train = price_scaler.inverse_transform(y_train_full)[:, 3]
        y_test = price_scaler.inverse_transform(y_test_full)[:, 3]
        
        # Plot results
        plot_predictions(
            np.concatenate([y_train, y_test]),
            np.concatenate([train_predictions, test_predictions]),
            train_size
        )
        
        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Save the model
        save_path = 'C:/Users/User/Desktop/repos/PMK_Course_work/saved_models'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model_path = os.path.join(save_path, 'bitcoin_model.h5')
        model.save(model_path)
        print(f"\nNN saved at: {model_path}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()