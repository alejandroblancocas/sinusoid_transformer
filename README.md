# Time Series Forecasting with GRU

Deep Learning project focused on the use of **GRU recurrent neural networks for time-series forecasting**, implemented with PyTorch.

The project is divided into two experiments. First, the model is trained on a synthetic sinusoidal signal, providing a simple environment to test its ability to learn and predict sequential patterns. The same approach is then extended to a real-world time series using historical **Apple (AAPL) stock prices** obtained with `yfinance`.

The objective was not to build a financial prediction tool, but to explore how recurrent neural networks behave when moving from a predictable synthetic sequence to a considerably more complex and noisy real-world dataset.

## Project contents

* GRU models implemented with PyTorch
* Synthetic sinusoidal time-series generation and prediction
* Multi-step forecasting of stock prices
* Historical AAPL data obtained with `yfinance`
* Train/test separation and sliding-window preprocessing
* GPU support with CUDA / MPS when available
* Visualization and comparison of predicted and real values

## Structure

```text
├── notebooks/
│   ├── main.ipynb
│   ├── sinusoide.ipynb
│   ├── stock_prediction.ipynb
│   └── gru_model.pt
├── src/
│   ├── constructor_sinu.py
│   └── mlp.py
├── requirements.txt
└── README.md
```

`main.ipynb` contains the complete experiment, while the other notebooks contain the individual development of each part.

## Technologies

**Python · PyTorch · NumPy · Matplotlib · yfinance · Jupyter**

## Running the project

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Then open the notebooks with Jupyter and run `main.ipynb` to reproduce the complete experiment.

## Results

The sinusoidal experiment shows that the GRU can learn and reproduce a simple and predictable temporal pattern with high accuracy.

The stock-price experiment illustrates the additional difficulty of forecasting real financial data. While the network is capable of learning patterns from historical values, its predictions also show the limitations of using past prices alone to forecast an inherently noisy and complex system.

> The stock forecasting section is an experimental application of recurrent neural networks to financial time series and is not intended as a financial forecasting or investment tool.
