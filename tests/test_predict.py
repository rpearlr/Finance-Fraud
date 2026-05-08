import requests
import json
import math

features = {
    "tx_id": "TX-94821",
    "amount_log": math.log(2400.50),
    "hour_of_day": 3,
    "amount_zscore": 4.1,
    "velocity_24tx": 8200.0,
    "V1": -3.2,
    "V2": 2.8,
    "V3": -10, "V4": 10, "V5": 0, "V6": 0, "V7": -5, "V8": 0, "V9": -5, "V10": -10,
    "V11": 5, "V12": -10, "V13": 0, "V14": -10, "V15": 0, "V16": -5, "V17": -10, "V18": 0, "V19": 0, "V20": 0,
    "V21": 0, "V22": 0, "V23": 0, "V24": 0, "V25": 0, "V26": 0, "V27": 0, "V28": 0
}

r = requests.post("http://localhost:8000/predict", json=features)
print(r.json())
