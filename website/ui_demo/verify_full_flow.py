
import requests
from requests.exceptions import RequestException
import sys
import json
import time

BASE_URL = "http://127.0.0.1:8000/api"

def log(msg, status="INFO"):
    print(f"[{status}] {msg}")

def fail(msg):
    log(msg, "FAIL")
    sys.exit(1)

def verify_predict():
    log("Verifying /predict...")
    payload = {
        "text": "The battery is amazing but the screen is terrible.",
        "msr_enabled": True,
        "msr_strength": 0.3
    }
    try:
        r = requests.post(f"{BASE_URL}/predict", json=payload, timeout=300)
        r.raise_for_status()
        data = r.json()
        
        # Check structure
        if "aspects" not in data: fail("Missing 'aspects' in predict response")
        if "conflict_prob" not in data: fail("Missing 'conflict_prob' in predict response")
        if not isinstance(data["aspects"], list): fail("'aspects' is not a list")
        
        log(f"Predict Response OK. Conflict Prob: {data['conflict_prob']}")
        return data
    except Exception as e:
        fail(f"Predict failed: {e}")

def verify_explain(text):
    log("Verifying /explain...")
    payload = {
        "text": text,
        "aspect": "all",
        "methods": ["ig"],
        "msr_enabled": True,
        "msr_strength": 0.3
    }
    try:
        r = requests.post(f"{BASE_URL}/explain", json=payload)
        r.raise_for_status()
        data = r.json()
        
        # Check structure
        if "ig_conflict" not in data: fail("Missing ig_conflict")
        if "aspects" not in data: fail("Missing aspects")
        
        log("Explain Response OK.")
    except Exception as e:
        fail(f"Explain failed: {e}")

def verify_metrics():
    log("Verifying /metrics...")
    try:
        r = requests.get(f"{BASE_URL}/metrics")
        r.raise_for_status()
        data = r.json()
        if "overall_macro_f1_4class" not in data: fail("Missing metrics fields")
        log("Metrics Response OK.")
    except Exception as e:
        fail(f"Metrics failed: {e}")

def verify_logs():
    log("Verifying /logs...")
    try:
        r = requests.get(f"{BASE_URL}/logs")
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list): fail("Logs should be a list")
        log(f"Logs Response OK. Items: {len(data)}")
    except Exception as e:
        fail(f"Logs failed: {e}")

def main():
    # Wait for server to potentiall start if just launched
    # But we assume it is running as user is waiting
    try:
        data = verify_predict()
        verify_explain("The battery is amazing but the screen is terrible.")
        verify_metrics()
        verify_logs()
        log("ALL CHECKS PASSED", "SUCCESS")
    except RequestException:
         fail("Could not connect to server. Is it running?")

if __name__ == "__main__":
    main()
