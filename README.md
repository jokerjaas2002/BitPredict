# CryptoPricePredictor

![Banner](https://via.placeholder.com/1200x300.png?text=CryptoPricePredictor)  
*A tool to predict cryptocurrency prices and suggest buy/sell actions using machine learning.*

---

## 📖 About
CryptoPricePredictor is a Python-based tool that predicts the next day's price for any cryptocurrency (e.g., Bitcoin, Ethereum, XRP, Theta) using historical data from [CoinGecko](https://www.coingecko.com/). It leverages a linear regression model and a simple moving average (SMA) to provide actionable suggestions: **Buy**, **Sell**, or **Hold**. Perfect for traders and developers interested in crypto market analysis.

## ✨ Features
- Predicts the next day's price for any CoinGecko-supported cryptocurrency.
- Provides **Buy**, **Sell**, or **Hold** suggestions based on price predictions and SMA.
- User-friendly: Enter the Coin ID (e.g., `bitcoin`, `ripple`, `theta-token`) to analyze your chosen crypto.
- Includes model accuracy metrics (R² score).
- Built with Python, pandas, scikit-learn, and requests.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/jokerjaas2002/CryptoPricePredictor.git
   cd CryptoPricePredictor