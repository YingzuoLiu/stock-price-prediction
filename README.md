# 使用Transformer进行股票价格预测

## 概述

本项目探索了使用基于Transformer的神经网络预测股票价格，特别针对苹果公司（AAPL）。项目展示了一个自定义Transformer模型的构建、训练和评估，该模型旨在处理金融时间序列数据。

## 代码与模型结构

### 主要特点

- **数据收集**：使用`yfinance`获取过去三年的历史股票数据。
- **特征选择**：包括关键属性，例如`Open`（开盘价）、`High`（最高价）、`Low`（最低价）、`Close`（收盘价）和`Volume`（成交量）。
- **序列生成**：利用过去90天的数据预测未来60天的收盘价格。
- **自定义Transformer编码器**：
  - 多头注意力机制捕捉复杂的时间依赖关系。
  - 前馈网络学习高级表示。
  - 层归一化和dropout用于正则化。

### 模型架构

Transformer模型包括以下部分：

- **输入形状**：接受形状为`(90, 5)`的序列（90天的5个特征）。
- **Transformer模块**：堆叠了四个Transformer编码器层，参数如下：
  - `head_size`（头大小）：64
  - `num_heads`（注意力头数）：4
  - `ff_dim`（前馈网络维度）：128
  - `dropout`（丢弃率）：0.1
- **注意力机制**：通过加权注意力聚合时间信息。
- **输出层**：一个全连接层预测收盘价。

## 训练配置

### 参数

- **批量大小**：64
- **学习率**：0.001（Adam优化器）
- **损失函数**：均方误差（MSE）
- **评价指标**：平均绝对误差（MAE）

### 训练策略

- 将数据分为训练集（80%）和验证集（20%）。
- 使用带有5轮耐心期的早停法防止过拟合。
- 通过dropout层和小学习率进行正则化。

### 避免过拟合

- 在Transformer层中应用dropout。
- 将Transformer模块数量限制为4个。
- 使用早停法监控验证性能。
- 使用`StandardScaler`对特征进行标准化，稳定训练过程中的梯度。

## 结果

### 训练指标

- **最终训练损失**：约0.097（MSE）
- **最终验证损失**：约0.118（MSE）
- **最终MAE**：约0.26


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


