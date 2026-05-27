# Impact of Dropout Patterns in Deep LSTM for Stock Price Prediction

This project evaluates five different dropout configurations in stacked LSTMs on Google stock price data.

## Requirements
Install dependencies with pip install -r requirements.txt

## Usage
- Place 	rain.csv and 	est.csv in the data/ folder.
- Run python src/train.py to train and evaluate models.
- Results and plots will be saved in esults/.

## Results
The ascending dropout pattern (0.2 → 0.5) achieved the best RMSE (24.8) and MAPE (6.95%).
