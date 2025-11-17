# Add this line at the very top of your file
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" # Disable oneDNN to help with stability/memory
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from flask import Flask, render_template, request, jsonify, url_for, Response
import pandas as pd
import numpy as np
import joblib
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError,MeanAbsoluteError

model_block = load_model("lstm_crop_yield_model.keras",compile=False)
scaler_block = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")
features_used = joblib.load("features.pkl")
    
with open("config.json", "r") as f:
    config = json.load(f)
    
seq_length = config["sequence_length"]
target_col = config["target_column"]
year_col = config["year_column"]
    
df = pd.read_csv("cleaned_data.csv")
crops = sorted(df['Crop'].unique())
blocks = sorted(df['Block_name'].unique())

X_dummy_block = np.zeros((1, seq_length, len(features_used) - 2))
crop_dummy = np.zeros((1, seq_length))
block_dummy = np.zeros((1, seq_length))

try:
    model_block.predict([crop_dummy, block_dummy, X_dummy_block], verbose=0)
    print("✅ Block Model Pre-Warmed Successfully.")
except Exception as e:
    print(f"Warning: Failed to pre-warm block model: {e}")

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/blockwise')
def blockwise():
    return render_template('blockwise.html')
@app.route('/districtwise')
def districtwise():
    return render_template('districtwise.html')
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')
@app.route("/blockwise/predict", methods=["POST"])
def predict():
    crop_name = request.form["Crop"]
    block_name = request.form["Block_name"]
    target_year = int(request.form["Year"])

    # Encode crop and block names
    crop_id = encoders["crop_encoder"].transform([crop_name])[0]
    block_id = encoders["block_encoder"].transform([block_name])[0]
    seq_len = config["sequence_length"]  # e.g. 5

    # Filter dataset for the given crop & block
    df_crop = df[(df["Crop"] == crop_name) & (df["Block_name"] == block_name)].sort_values("Year")

    # Compute last seq_len years before target
    input_years = list(range(target_year - seq_len, target_year))
    last_5_years = df_crop[df_crop["Year"].isin(input_years)].copy()

    if len(last_5_years) < seq_len:
        return jsonify({
            "error": f"Not enough data to predict {target_year}. Need {seq_len} years before it."
        })

    print(f"🔹 Using data from years: {last_5_years['Year'].tolist()} for predicting {target_year}")

    # Add encoded IDs as columns (so they appear in features_used)
    last_5_years["Crop_ID"] = crop_id
    last_5_years["Block_ID"] = block_id

    # Select only required features
    X_input = last_5_years[features_used].values

    # Scale numeric columns (same as during training)
    X_scaled = X_input.copy()
    X_scaled[:, 2:] = scaler_block.transform(X_input[:, 2:])  # assuming first 2 are categorical IDs

    # Split inputs for model
    crop_seq = X_scaled[:, 0].astype(int).reshape(1, seq_len)
    block_seq = X_scaled[:, 1].astype(int).reshape(1, seq_len)
    num_seq = X_scaled[:, 2:].reshape(1, seq_len, len(features_used) - 2)

    # Predict yield
    predicted_yield = model_block.predict([crop_seq, block_seq, num_seq])[0][0]

    return jsonify({
        "crop": crop_name,
        "block": block_name,
        "target_year": target_year,
        "predicted_yield": float(predicted_yield),
        "used_years": input_years
    })
MODEL_PATH = "crop_yield_model.keras"
SCALER_PATH = "feature_scaler1.save"
ENCODING_PATH = "encoding_info.json"
DATA_PATH = "df_disrtict.csv"
model_district = load_model(MODEL_PATH, compile=False)
scaler_district = joblib.load(SCALER_PATH)

with open(ENCODING_PATH, "r") as f:
    encoding_info = json.load(f)

feature_cols = encoding_info["feature_columns"]

# Load dataset
df_all = pd.read_csv(DATA_PATH)
crops = sorted(df_all["Crop"].unique())
districts = sorted(df_all["District"].unique())
years = sorted(df_all["Year"].unique()) + [max(df_all["Year"]) + 1]
@app.route("/districtwise/predict", methods=["POST"])
def predict_district():
    crop_name = request.form["Crop"]
    district_name = request.form["District"]
    target_year = int(request.form["Year"])
    seq_len = 5  # same as before

    # Filter by crop and district
    df_crop = df_all[(df_all["Crop"] == crop_name) & (df_all["District"] == district_name)].sort_values("Year")

    # Select last seq_len years before the target year
    input_years = list(range(target_year - seq_len, target_year))
    df_recent = df_crop[df_crop["Year"].isin(input_years)].copy()

    if len(df_recent) < seq_len:
        return jsonify({
            "error": f"Not enough past data for {crop_name} - {district_name} to predict {target_year}."
        })

    # One-hot encode Crop and District
    df_recent = pd.get_dummies(df_recent, columns=["Crop", "District"])

    # Add any missing columns
    for col in feature_cols:
        if col not in df_recent.columns:
            df_recent[col] = 0

    # Reorder columns
    df_recent = df_recent[feature_cols]

    # Scale numeric features
    df_scaled = scaler_district.transform(df_recent)

    # Reshape for LSTM
    X_input = np.expand_dims(df_scaled, axis=0)

    # Predict
    predicted_yield = model_district.predict(X_input)[0][0]
    y_pred_original = np.expm1(predicted_yield)

    return jsonify({
        "crop": crop_name,
        "district": district_name,
        "target_year": target_year,
        "predicted_yield": float(y_pred_original),
        "used_years": df_recent.index.tolist()
    })


if __name__ == '__main__':
    # Start Flask app
    app.run()









