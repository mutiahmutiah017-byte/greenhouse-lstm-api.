"""
app.py
======
API kecil yang menjalankan model LSTM yang sudah dilatih (train_lstm.py)
untuk memprediksi suhu & kelembapan langkah berikutnya, lalu diambil
oleh website (script.js) lewat fetch().

Alur (sesuai diagram: Model LSTM -> Grafik website):
1. Ambil WINDOW pembacaan terakhir dari Firebase
2. Normalisasi pakai scaler yang sama dengan saat training
3. model.predict() -> hasil skala 0-1
4. Kembalikan ke skala asli (inverse_transform)
5. Kirim sebagai JSON ke frontend

Jalankan:
    pip install -r requirements.txt
    python app.py
Lalu deploy ke layanan seperti Render/Railway/PythonAnywhere supaya
website (yang berjalan di browser publik) bisa mengaksesnya.
"""

import json
import os
import numpy as np
import requests
import joblib
from flask import Flask, jsonify
from flask_cors import CORS
from tensorflow import keras

FB_HOST = "greenhouse-2c990-default-rtdb.asia-southeast1.firebasedatabase.app"

# File riwayat training (accuracy & loss per-epoch) hasil train_lstm_classifier.py.
# Dipakai untuk kurva accuracy/loss di dashboard website (revisi penguji poin 6).
TRAINING_HISTORY_FILES = {
    "classifier": "training_history.json",
}

app = Flask(__name__)
CORS(app)  # izinkan website (domain berbeda) memanggil API ini

model = keras.models.load_model("lstm_model.keras")
scaler = joblib.load("scaler.save")
meta = joblib.load("meta.save")
WINDOW = meta["window"]
FEATURES = meta["features"]


def fetch_recent_readings(plant, limit=30):
    """Ambil beberapa log tanggal terakhir lalu gabungkan jadi satu deret waktu."""
    dates_url = f"https://{FB_HOST}/greenhouse/logs.json?shallow=true"
    r = requests.get(dates_url, timeout=20)
    r.raise_for_status()
    dates = sorted((r.json() or {}).keys())[-3:]  # 3 tanggal terakhir cukup untuk WINDOW=6

    rows = []
    for tanggal in dates:
        day_url = f"https://{FB_HOST}/greenhouse/logs/{tanggal}.json"
        rd = requests.get(day_url, timeout=20).json() or {}
        for jam, val in rd.items():
            rec = val if isinstance(val, dict) and "suhu" in val else (val or {}).get(plant)
            if rec and "suhu" in rec and "kelembapan" in rec:
                rows.append({
                    "timestamp": f"{tanggal} {jam}",
                    "suhu": float(rec["suhu"]),
                    "kelembapan": float(rec["kelembapan"]),
                })
    rows.sort(key=lambda x: x["timestamp"])
    return rows[-WINDOW:]


def classify_from_values(suhu, kelembapan):
    """Sama seperti classify() di script.js, dipakai untuk melabeli hasil prediksi."""
    if suhu > 35:
        return "Terlalu Panas", "Segera buka ventilasi dan kurangi sumber panas"
    if 27 <= suhu <= 35:
        return "Agak Panas", "Buka ventilasi, tambah sirkulasi udara"
    if kelembapan > 90:
        return "Terlalu Lembap", "Buka ventilasi untuk mengurangi kelembapan"
    if kelembapan < 60:
        return "Kurang Lembap", "Tambah penyiraman atau kabut air"
    return "Optimal", "Kondisi ideal"


FORECAST_STEPS = 6  # jumlah langkah ke depan yang diprediksi, membentuk pola tren


def forecast_multi_step(values, steps):
    """
    Prediksi berulang (recursive forecasting): hasil prediksi langkah 1
    dipakai sebagai bagian input untuk memprediksi langkah 2, dst.
    Ini yang membuat grafiknya menunjukkan POLA/TREN, bukan cuma 1 titik.
    """
    window = values.copy()  # sudah dalam skala 0-1
    preds = []
    for _ in range(steps):
        X = window.reshape(1, WINDOW, len(FEATURES))
        next_scaled = model.predict(X, verbose=0)[0]
        preds.append(next_scaled)
        window = np.vstack([window[1:], next_scaled])
    return np.array(preds)


@app.route("/predict/<plant>")
def predict(plant):
    readings = fetch_recent_readings(plant)
    if len(readings) < WINDOW:
        return jsonify({
            "error": f"Data belum cukup. Butuh {WINDOW} pembacaan terakhir, baru ada {len(readings)}."
        }), 400

    values = np.array([[r["suhu"], r["kelembapan"]] for r in readings])
    scaled = scaler.transform(values)

    preds_scaled = forecast_multi_step(scaled, FORECAST_STEPS)
    preds = scaler.inverse_transform(preds_scaled)

    forecast = [
        {"langkah": i + 1, "suhu": round(float(p[0]), 2), "kelembapan": round(float(p[1]), 2)}
        for i, p in enumerate(preds)
    ]
    suhu_pred, kelembapan_pred = forecast[0]["suhu"], forecast[0]["kelembapan"]
    kelas, aksi = classify_from_values(suhu_pred, kelembapan_pred)

    return jsonify({
        "plant": plant,
        "input_window": readings,
        "prediksi": {
            "suhu": suhu_pred,
            "kelembapan": kelembapan_pred,
            "klasifikasi": kelas,
            "rekomendasi": aksi,
        },
        "forecast": forecast,  # daftar 6 langkah ke depan, untuk digambar sebagai pola/tren
    })


@app.route("/training-history/<model_name>")
def training_history(model_name):
    """
    Kembalikan riwayat accuracy & loss (train + validation) per-epoch dari
    proses pelatihan model LSTM klasifikasi, untuk digambar sebagai kurva
    di dashboard website (Chart.js, lihat script.js -> loadTrainingHistory()).

    File training_history.json dibuat otomatis oleh
    train_lstm_classifier.py setiap kali model dilatih ulang -- jalankan
    ulang script itu (python train_lstm_classifier.py) untuk memperbarui
    kurva ini setelah tuning learning rate / window / dsb.
    """
    path = TRAINING_HISTORY_FILES.get(model_name)
    if not path or not os.path.exists(path):
        return jsonify({
            "error": (
                f"Riwayat training untuk '{model_name}' belum ada. "
                "Jalankan train_lstm_classifier.py terlebih dahulu supaya "
                "training_history.json terbentuk."
            )
        }), 404
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return jsonify(data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
