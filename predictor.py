import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from ta.momentum import RSIIndicator
from ta.trend import MACD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Función para obtener la lista de criptomonedas soportadas por CoinGecko
def get_supported_coins():
    url = 'https://api.coingecko.com/api/v3/coins/list'
    response = requests.get(url)
    if response.status_code == 200:
        coins = response.json()
        return {coin['id']: coin['name'] for coin in coins}
    else:
        raise Exception("Error al obtener la lista de criptomonedas")

# Mapear CoinGecko IDs a símbolos de Binance
coin_to_binance = {
    'bitcoin': 'BTCUSDT',
    'ethereum': 'ETHUSDT',
    'solana': 'SOLUSDT',
    'ripple': 'XRPUSDT',
    'theta-token': 'THETAUSDT'
}

# Función para obtener datos históricos de Binance
def get_binance_data(symbol, interval='1h', days=30):
    url = 'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': days * 24  # Horas en el período
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"Error al obtener datos para {symbol}")
    data = response.json()
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['price'] = df['close'].astype(float)
    return df[['date', 'price']]

# Calcular indicadores (SMA, RSI, MACD)
def calculate_indicators(df, sma_window=7, rsi_window=14):
    df['sma'] = df['price'].rolling(window=sma_window).mean()
    rsi = RSIIndicator(df['price'], window=rsi_window)
    df['rsi'] = rsi.rsi()
    macd = MACD(df['price'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    return df

# Preparar datos para LSTM
def prepare_lstm_data(df, lookback=7):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['price']].values)
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i])
        y.append(scaled_data[i])
    return np.array(X), np.array(y), scaler

# Entrenar modelo LSTM y predecir
def train_and_predict(df, hours_ahead=24):
    X, y, scaler = prepare_lstm_data(df)
    if len(X) < 10:  # Evitar errores con pocos datos
        return df['price'].iloc[-1], 0.0
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(7, 1), return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    last_sequence = X[-1:]
    predicted_scaled = model.predict(last_sequence)
    predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]
    y_test_pred = model.predict(X_test)
    ss_tot = np.sum((y_test - np.mean(y_test))**2)
    ss_res = np.sum((y_test - y_test_pred)**2)
    r2_score = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return predicted_price, r2_score

# Sugerir acción de compra/venta
def suggest_action(current_price, predicted_price, sma, rsi, macd, macd_signal, threshold=0.02):
    price_change = (predicted_price - current_price) / current_price
    macd_bullish = macd > macd_signal
    if price_change > threshold and current_price < sma and rsi < 30 and macd_bullish:
        return "Comprar", f"El precio subirá más del {threshold*100:.1f}%, está bajo la SMA, RSI indica sobreventa (<30), y MACD es alcista."
    elif price_change < -threshold and current_price > sma and rsi > 70 and not macd_bullish:
        return "Vender", f"El precio bajará más del {threshold*100:.1f}%, está sobre la SMA, RSI indica sobrecompra (>70), y MACD es bajista."
    else:
        return "Mantener", f"El cambio es menor al {threshold*100:.1f}% o no hay señal clara (RSI: {rsi:.2f}, MACD: {macd:.2f})."

# Backtesting
def backtest(df, threshold=0.02):
    balance = 1000  # Saldo inicial en USD
    position = 0  # Cantidad de cripto
    trades = []
    for i in range(14, len(df)-24):
        current_price = df['price'].iloc[i]
        sma = df['sma'].iloc[i]
        rsi = df['rsi'].iloc[i]
        macd = df['macd'].iloc[i]
        macd_signal = df['macd_signal'].iloc[i]
        predicted_price = train_and_predict(df.iloc[:i+1])[0]
        action, _ = suggest_action(current_price, predicted_price, sma, rsi, macd, macd_signal, threshold)
        if action == "Comprar" and balance > 0:
            position = balance / current_price
            balance = 0
            trades.append(f"Compra a ${current_price:.2f}")
        elif action == "Vender" and position > 0:
            balance = position * current_price
            position = 0
            trades.append(f"Venta a ${current_price:.2f}")
    final_value = balance + position * df['price'].iloc[-1]
    return final_value, trades

# Función principal
def main():
    # Obtener lista de criptomonedas soportadas
    supported_coins = get_supported_coins()
    
    # Solicitar al usuario que elija una criptomoneda
    while True:
        coin_id = input("Ingresa el ID de la criptomoneda (ej. bitcoin, ethereum, solana): ").lower().strip()
        if coin_id in supported_coins and coin_id in coin_to_binance:
            coin_name = supported_coins[coin_id]
            binance_symbol = coin_to_binance[coin_id]
            print(f"Criptomoneda seleccionada: {coin_name} ({coin_id}) - Par en Binance: {binance_symbol}")
            break
        else:
            print("ID de criptomoneda no válido o no soportado en Binance. Intenta con bitcoin, ethereum, solana, ripple, theta-token.")

    try:
        # Obtener datos históricos de Binance
        df = get_binance_data(binance_symbol)
        
        # Calcular indicadores
        df = calculate_indicators(df)
        
        # Backtesting
        final_value, trades = backtest(df)
        
        # Entrenar modelo y predecir
        predicted_price, score = train_and_predict(df)
        
        # Obtener precios actuales, SMA, RSI y MACD
        current_price = df['price'].iloc[-1]
        current_sma = df['sma'].iloc[-1]
        current_rsi = df['rsi'].iloc[-1]
        current_macd = df['macd'].iloc[-1]
        current_macd_signal = df['macd_signal'].iloc[-1]
        
        # Obtener sugerencia
        action, reason = suggest_action(current_price, predicted_price, current_sma, current_rsi, current_macd, current_macd_signal)
        
        # Imprimir resultados
        last_date = df['date'].iloc[-1]
        prediction_date = last_date + timedelta(days=1)
        print(f"\nResultados para {coin_name} ({coin_id}):")
        print(f"Último precio conocido: ${current_price:.2f}")
        print(f"Media móvil (7 horas): ${current_sma:.2f}")
        print(f"RSI (14 horas): {current_rsi:.2f}")
        print(f"MACD: {current_macd:.2f}, Signal: {current_macd_signal:.2f}")
        print(f"Predicción para {prediction_date.date()}: ${predicted_price:.2f}")
        print(f"R² Score del modelo: {score:.4f}")
        print(f"Sugerencia: {action}")
        print(f"Razón: {reason}")
        print(f"\nBacktesting (saldo inicial $1000):")
        print(f"Valor final: ${final_value:.2f}")
        print(f"Trades realizados: {trades if trades else 'Ninguno'}")
    
    except Exception as e:
        print(f"Error: {str(e)}")

# Ejecutar
if __name__ == "__main__":
    main()