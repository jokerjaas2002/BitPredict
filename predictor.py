import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

# Función para obtener la lista de criptomonedas soportadas por CoinGecko
def get_supported_coins():
    url = 'https://api.coingecko.com/api/v3/coins/list'
    response = requests.get(url)
    if response.status_code == 200:
        coins = response.json()
        return {coin['id']: coin['name'] for coin in coins}
    else:
        raise Exception("Error al obtener la lista de criptomonedas")

# Función para obtener datos históricos de CoinGecko
def get_historical_data(coin_id, days=30):
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart'
    params = {'vs_currency': 'usd', 'days': days, 'interval': 'daily'}
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"Error al obtener datos para {coin_id}")
    data = response.json()
    
    prices = data['prices']
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.drop('timestamp', axis=1, inplace=True)
    return df

# Calcular media móvil simple (SMA)
def calculate_sma(df, window=7):
    df['sma'] = df['price'].rolling(window=window).mean()
    return df

# Preparar datos para el modelo
def prepare_data(df):
    df['days'] = (df['date'] - df['date'].min()).dt.days
    X = df[['days']].values
    y = df['price'].values
    return X, y

# Entrenar modelo y predecir
def train_and_predict(df, days_ahead=1):
    X, y = prepare_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    last_day = X[-1][0]
    future_day = last_day + days_ahead
    predicted_price = model.predict([[future_day]])[0]
    
    score = model.score(X_test, y_test)
    
    return predicted_price, score

# Sugerir acción de compra/venta
def suggest_action(current_price, predicted_price, sma, threshold=0.02):
    price_change = (predicted_price - current_price) / current_price
    
    if price_change > threshold and current_price < sma:
        return "Comprar", f"El precio predicho subirá más del {threshold*100:.1f}% y está por debajo de la media móvil."
    elif price_change < -threshold and current_price > sma:
        return "Vender", f"El precio predicho bajará más del {threshold*100:.1f}% y está por encima de la media móvil."
    else:
        return "Mantener", f"El cambio predicho es menor al {threshold*100:.1f}% o no hay una señal clara respecto a la media móvil."

# Función principal
def main():
    # Obtener lista de criptomonedas soportadas
    supported_coins = get_supported_coins()
    
    # Solicitar al usuario que elija una criptomoneda
    while True:
        coin_id = input("Ingresa el ID de la criptomoneda (ej. bitcoin, ethereum, solana): ").lower().strip()
        if coin_id in supported_coins:
            coin_name = supported_coins[coin_id]
            print(f"Criptomoneda seleccionada: {coin_name} ({coin_id})")
            break
        else:
            print("ID de criptomoneda no válido. Por favor, intenta de nuevo.")
    
    try:
        # Obtener datos históricos
        df = get_historical_data(coin_id)
        
        # Calcular media móvil
        df = calculate_sma(df)
        
        # Entrenar modelo y predecir
        predicted_price, score = train_and_predict(df)
        
        # Obtener precios actuales y SMA
        current_price = df['price'].iloc[-1]
        current_sma = df['sma'].iloc[-1]
        
        # Obtener sugerencia
        action, reason = suggest_action(current_price, predicted_price, current_sma)
        
        # Imprimir resultados
        last_date = df['date'].iloc[-1]
        prediction_date = last_date + timedelta(days=1)
        print(f"\nResultados para {coin_name} ({coin_id}):")
        print(f"Último precio conocido: ${current_price:.2f}")
        print(f"Media móvil (7 días): ${current_sma:.2f}")
        print(f"Predicción para {prediction_date.date()}: ${predicted_price:.2f}")
        print(f"R² Score del modelo: {score:.4f}")
        print(f"Sugerencia: {action}")
        print(f"Razón: {reason}")
    
    except Exception as e:
        print(f"Error: {str(e)}")

# Ejecutar
if __name__ == "__main__":
    main()