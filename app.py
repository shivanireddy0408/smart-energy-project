from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import datetime

app = Flask(__name__)

# Load dataset and train model (only once at startup)
def train_model():
    df = pd.read_csv('energy_con_dataset.txt', sep=';', low_memory=False, na_values='?')
    df.dropna(inplace=True)
    
    # Combine date and time, convert to datetime
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
    df.set_index('Datetime', inplace=True)

    # Convert to float
    df['Global_active_power'] = df['Global_active_power'].astype(float)

    # Resample data by hour
    df_hourly = df['Global_active_power'].resample('h').mean().ffill()

    
    # Create lag features
    df_hourly = pd.DataFrame(df_hourly)
    df_hourly['Prev_Hour'] = df_hourly['Global_active_power'].shift(1)
    df_hourly.dropna(inplace=True)

    # Prepare features and target
    X = df_hourly[['Prev_Hour']]
    y = df_hourly['Global_active_power']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = LinearRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, 'model/energy_model.pkl')
    print("Model trained and saved.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_value = float(request.form['prev_hour'])
        model = joblib.load('model/energy_model.pkl')
        prediction = model.predict([[input_value]])[0]
        return jsonify({'prediction': round(prediction, 3)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
   # train_model()
    app.run(debug=True)

