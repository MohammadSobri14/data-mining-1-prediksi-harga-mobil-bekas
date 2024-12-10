from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Memuat model yang sudah disimpan
model = joblib.load('model/model_estimasi_harga_mobil.pkl')

@app.route('/')
def index():
    return "Welcome to the Price Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    # Mendapatkan data JSON dari request
    data = request.get_json()

    # Misalnya, data yang diterima adalah list fitur: ['prod_year', 'mileage', 'manufacturer', 'age', 'color']
    features = data['features']

    # Lakukan transformasi atau preprocessing jika diperlukan
    # Contoh: scaling data, encoding kategori, dll.
    # Misalnya, jika menggunakan scaler atau preprocessing, lakukan di sini.
    
    # Prediksi harga menggunakan model
    prediction = model.predict([features])  # Sesuaikan bentuk input dengan model

    # Mengembalikan hasil prediksi sebagai JSON
    return jsonify({'predicted_price': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
