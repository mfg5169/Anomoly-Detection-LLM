import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch, PyTorchModel
from sagemaker.xgboost.estimator import XGBoost

role = sagemaker.get_execution_role()  # Automatically retrieves the IAM role
sagemaker_session = sagemaker.Session()
s3_bucket = sagemaker_session.default_bucket()

def fetch_data(ticker="TSLA", period="2y", interval="1h"):
    data = yf.download(ticker, period=period, interval=interval)
    return data

def compute_features(data):
    data["SMA_10"] = data["Close"].rolling(window=10).mean()
    data["SMA_50"] = data["Close"].rolling(window=50).mean()
    data["RSI"] = compute_rsi(data["Close"])
    data["Volatility"] = data["Close"].pct_change().rolling(10).std()
    return data.dropna()

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def label_data(data, short_term=10, long_term=50, threshold=0.05):
    data["Future_Return"] = data["Close"].pct_change(short_term).shift(-short_term)
    data["Target"] = np.where(data["Future_Return"] > threshold, 1, 0)  # 1 = Good Short-Term Buy
    return data.dropna()

def prepare_lstm_data(data, sequence_length=10):
    features = ["SMA_10", "SMA_50", "RSI", "Volatility"]
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[features].iloc[i:i+sequence_length].values)
        y.append(data["Target"].iloc[i+sequence_length])
    return np.array(X), np.array(y)

def extract_lstm_features(lstm_model, data, sequence_length=10):
    features = ["SMA_10", "SMA_50", "RSI", "Volatility"]
    X = []
    for i in range(len(data) - sequence_length):
        X.append(data[features].iloc[i:i+sequence_length].values)
    X = np.array(X)
    lstm_model.eval()
    with torch.no_grad():
        lstm_features = lstm_model(torch.tensor(X, dtype=torch.float32)).numpy()
    return lstm_features


def train_lstm_sagemaker(bucket, location):
    lstm_estimator = PyTorch(
        entry_point="train_lstm.py",
        role=role,
        instance_count=1,
        instance_type="ml.m5.2xlarge",
        framework_version="1.9",
        py_version="py38",
        script_mode=True,
        hyperparameters={"epochs": 1, "batch-size": 32},
        environment={
        'PYTHONUNBUFFERED': '1'
            },
        source_dir='.',
    )
    lstm_estimator.fit({"training" :f"s3://{bucket}/{location}"}, wait=True, logs='All')




    pytorch_model = PyTorchModel(
        model_data=lstm_estimator.model_data,
        role=role,
        entry_point="inference.py",
        source_dir=".",
        framework_version="1.13",
        py_version="py39"
    )
    
    predictor = pytorch_model.deploy(instance_type="ml.m5.large", initial_instance_count=1, endpoint_name='Tesla-Trial')

    return predictor

def train_xgboost_sagemaker(X_train, y_train):
    xgb_estimator = XGBoost(
        entry_point="train_xgboost.py",
        role=role,
        instance_count=1,
        instance_type="ml.m5.2xlarge",
        framework_version="1.3-1",
        py_version="py3",
        script_mode=True
    )
    xgb_estimator.fit("s3://your-bucket/xgb-training-data")
    return xgb_estimator

def preprocesss_data(ticker="TSLA", period="2y", interval="1h"):
    data = fetch_data(ticker, period, interval)
    data = compute_features(data)
    data = label_data(data)
    X, y = prepare_lstm_data(data)
    return X, y, data


X, y = prepare_lstm_data(data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import io

def save_to_s3(X_train, X_test, y_train, y_test, bucket_name, prefix):
    """
    Save train-test data to S3 bucket
    
    Parameters:
    - X_train, X_test, y_train, y_test: numpy arrays
    - bucket_name: name of S3 bucket
    - prefix: folder path in the bucket
    """
    s3_client = boto3.client('s3')
    
    data_dict = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    
    for name, array in data_dict.items():
        buffer = io.BytesIO()
        np.save(buffer, array)
        buffer.seek(0)
        
        key = f"{prefix}/{name}.npy"
        s3_client.upload_fileobj(buffer, bucket_name, key)
        print(f"Saved {name} to s3://{bucket_name}/{key}")


bucket_name = s3_bucket  
prefix = 'lstm-data'  


save_to_s3(X_train, X_test, y_train, y_test, bucket_name, prefix)


lstm_model = train_lstm_sagemaker(bucket_name, prefix)

