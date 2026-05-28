

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**A systematic experiment to assess how different dropout configurations across stacked LSTM layers affect prediction smoothness, trend tracking, and pointwise accuracy on Google stock price data.**

## Research Question

**How does the placement and magnitude of dropout across four stacked LSTM layers influence the trade-off between smooth trend following and accurate local price prediction?**

## Hypotheses

- **H1:** A gradual increase in dropout from early to deep layers (ascending pattern) will preserve short-term details while providing sufficient regularization, yielding the best balance.
- **H2:** Strong dropout early in the network (descending or severe bottleneck) will oversmooth predictions, causing underfitting and high pointwise error.

## Key Findings

1. **Ascending dropout (0.2→0.5)** delivered the best overall performance – lowest RMSE (24.81), MAE (20.74), and MAPE (~7%). It captured both fine-grained movements and broad trends.
2. **Descending dropout (0.5→0.2) and the extra-LSTM architecture both underfit severely.** Starting with heavy dropout destroyed early signal; adding excessive depth with severe dropout blocked gradient flow.
3. **Hourglass and internal ascending bottleneck** showed improved trend detection but lacked pointwise precision.
4. **The placement of dropout matters more than its average value.** The pattern determines how information is compressed across the hierarchy.

## Results

![Comparison Curves](results/comparison_curves.png)
*Figure 1: Real price vs. predictions from all five dropout configurations.*

![Metrics Bar Chart](results/metrics_barchart.png)
*Figure 2: Quantitative comparison using RMSE, MAE, MAPE, and Directional Accuracy.*

## How to Reproduce

1. Clone the repository:
   ```bash
   git clone https://github.com/BenyaminMahdavifar/stock-price-prediction.git
   cd stock-price-prediction
