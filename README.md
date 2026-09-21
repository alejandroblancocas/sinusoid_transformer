# Sinusoid & Stock Prediction with GRU

Small deep learning project exploring the use of **GRU recurrent neural networks for time-series forecasting** with PyTorch.

The project started with a simple synthetic sinusoidal signal to test the model's ability to learn temporal patterns. The same idea was later extended to a more challenging problem using historical **Apple (AAPL) stock data** obtained through `yfinance`.

## What's included

* GRU model implemented with PyTorch
* Custom tensor-based batch loader
* One-step sinusoidal forecasting
* Multi-step stock price forecasting using log returns
* CUDA / MPS support when available
* Visualization of predictions against real data

## Structure

```text
├── notebooks/
│   ├── sinusoide.ipynb
│   ├── stock_prediction.ipynb
│   └── gru_model.pt
├── src/
│   ├── constructor_sinu.py
│   └── mlp.py
├── requirements.txt
└── README.md
```

## Technologies

**Python · PyTorch · NumPy · Matplotlib · yfinance**

## Running the project

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The experiments can then be run from the notebooks directory.

> The stock prediction notebook is an experimental application of recurrent neural networks to financial time series and is not intended as a financial forecasting tool.
