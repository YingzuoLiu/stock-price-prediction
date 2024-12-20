# Transformer for Stock Price Prediction

## Overview
This project explores the use of a Transformer-based neural network to predict stock prices, specifically targeting Apple Inc. (AAPL). It demonstrates the construction, training, and evaluation of a custom Transformer model designed to process sequential financial data.

## Code and Model Structure
### Key Features
- **Data Collection**: Utilized `yfinance` to fetch historical stock data for the past three years.
- **Feature Selection**: Included key attributes such as `Open`, `High`, `Low`, `Close`, and `Volume`.
- **Sequence Creation**: Used 90 past days of data to predict the closing price for the next 60 days.
- **Custom Transformer Encoder**:
  - Multi-head attention mechanism for capturing complex temporal dependencies.
  - Feedforward layers for learning higher-level representations.
  - Layer normalization and dropout for regularization.

### Model Architecture
The Transformer model includes the following:
- **Input Shape**: Accepts sequences of shape `(90, 5)` (90 days of 5 features).
- **Transformer Blocks**: Stacked four Transformer Encoder layers with the following parameters:
  - `head_size`: 64
  - `num_heads`: 4
  - `ff_dim`: 128
  - `dropout`: 0.1
- **Attention Mechanism**: Aggregates temporal information through weighted attention.
- **Output Layer**: A dense layer to predict the closing price.

## Training Configuration
### Parameters
- **Batch Size**: 64
- **Learning Rate**: 0.001 (Adam optimizer)
- **Loss Function**: Mean Squared Error (MSE)
- **Metrics**: Mean Absolute Error (MAE)

### Training Strategy
- Split the data into training (80%) and validation (20%) sets.
- Utilized early stopping with a patience of 5 epochs to prevent overfitting.
- Regularization through dropout layers and small learning rates.

### Avoiding Overfitting
- Applied dropout in Transformer layers.
- Limited the number of Transformer blocks to 4.
- Early stopping to monitor validation performance.
- Scaled features using `StandardScaler` to stabilize gradients during training.

## Results
### Training Metrics
- **Final Training Loss**: ~0.097 (MSE)
- **Final Validation Loss**: ~0.118 (MSE)
- **Final MAE**: ~0.26


