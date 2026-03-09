"""
Quick API test for ClearView backend (port 8000).
Tests: /health, /predict, /explain, /analyze (all-aspects)
"""
import json, time, sys
try:
    import requests
except ImportError:
    print("[ERROR] requests not installed: pip install requests")
    sys.exit(1)

BASE = "http://localhost:8000"
SEP  = "=" * 60

def test(label, method, url, **kwargs):
    try:
        t0  = time.time()
        r   = method(url, timeout=60, **kwargs)
        ms  = (time.time() - t0) * 1000
        ok  = 200 <= r.status_code < 300
        tag = "[OK ]" if ok else "[ERR]"
        print(f"{tag} {label:40s}  {r.status_code}  {ms:.0f}ms")
        return r if ok else None
    except requests.exceptions.ConnectionError:
        print(f"[CONN] {label} -- could not connect. Is backend running on port 8000?")
        return None
    except Exception as e:
        print(f"[FAIL] {label} -- {e}")
        return None

print(SEP)
print("  ClearView API Test Suite")
print(SEP)

# 1. Health check
print("\n[1] Health Check")
r = test("GET /", requests.get, f"{BASE}/")
if r:
    try:
        data = r.json()
        print(f"     status   : {data.get('status')}")
        print(f"     model    : {data.get('model_loaded')}")
        print(f"     device   : {data.get('device')}")
        print(f"     aspects  : {data.get('aspects')}")
    except Exception:
        print(f"     raw: {r.text[:300]}")

# Also try /health
r2 = test("GET /health", requests.get, f"{BASE}/health")
if r2:
    try:
        print(f"     /health: {r2.json()}")
    except Exception:
        print(f"     /health raw: {r2.text[:300]}")

# 2. Single aspect prediction
print("\n[2] Single-Aspect Prediction")
REVIEW = "I love the colour of this lipstick but the smell is absolutely awful and the packaging broke."

for aspect, expected in [("colour", "positive"), ("smell", "negative"), ("packing", "negative")]:
    r = test(f"POST /predict  [{aspect}]", requests.post,
             f"{BASE}/predict", json={"text": REVIEW, "aspect": aspect})
    if r:
        data = r.json()
        pred = data.get('sentiment') or data.get('label') or data.get('predicted_class') or str(data)[:50]
        conf = data.get('confidence', '?')
        ok   = pred == expected
        print(f"     {aspect:12} -> {pred:10} conf={conf}  {'CORRECT' if ok else f'WRONG (expected {expected})'}")
        if aspect == "colour":  # print all keys first time
            print(f"     Response keys: {list(data.keys())}")

# 3. Mixed sentiment resolution
print("\n[3] Mixed Sentiment Resolution")
MIXED = "The colour is beautiful but the smell is absolutely awful."
results = {}
for aspect, expected in [("colour", "positive"), ("smell", "negative")]:
    r = test(f"POST /predict  [{aspect}]", requests.post,
             f"{BASE}/predict", json={"text": MIXED, "aspect": aspect})
    if r:
        data = r.json()
        pred = data.get('sentiment') or data.get('label') or "?"
        results[aspect] = pred
        ok = pred == expected
        print(f"     [{('PASS' if ok else 'FAIL')}] {aspect} -> {pred}  (expected {expected})")
if len(results) == 2:
    t = "PASS" if results.get("colour")=="positive" and results.get("smell")=="negative" else "FAIL"
    print(f"     Overall MSR: {t}")

# 4. All-aspects / analyze endpoint
print("\n[4] All-Aspects Prediction")
found = False
for ep in ["/predict_all", "/analyze", "/analyze_all", "/predict/all"]:
    r = test(f"POST {ep}", requests.post, f"{BASE}{ep}", json={"text": REVIEW})
    if r:
        found = True
        data  = r.json()
        preds = data.get("predictions") or data.get("aspects") or \
                (data if isinstance(data, list) else [])
        print(f"     endpoint: {ep}")
        print(f"     aspects returned: {len(preds)}")
        for p in preds:
            name = p.get("name") or p.get("aspect","?")
            sent = p.get("sentiment") or p.get("label","?")
            conf = p.get("confidence", "?")
            print(f"       {name:<18} -> {sent}  conf={conf}")
        break
if not found:
    print("     [WARN] No multi-aspect endpoint found")

# 5. Edge cases
print("\n[5] Edge Cases")
edge_cases = [
    ("Short review 'Great!'",        {"text": "Great!", "aspect": "colour"}),
    ("Single word 'Terrible.'",      {"text": "Terrible.", "aspect": "smell"}),
    ("200-char review",              {"text": "The colour is absolutely stunning. The texture is greasy and heavy. The smell is nice but the price is too high. Shipping was quick but packaging cracked.", "aspect": "texture"}),
]
for label, payload in edge_cases:
    r = test(f"POST /predict  [{label}]", requests.post, f"{BASE}/predict", json=payload)
    if r:
        data = r.json()
        pred = data.get('sentiment') or data.get('label') or "?"
        print(f"     -> {pred}")

# 6. Explain endpoint
print("\n[6] Explain / XAI Endpoint")
found = False
for ep in ["/explain", "/explain_attention", "/explain_lime", "/xai"]:
    r = test(f"POST {ep}", requests.post, f"{BASE}{ep}",
             json={"text": REVIEW, "aspect": "colour", "method": "attention"})
    if r:
        found = True
        data = r.json()
        print(f"     endpoint: {ep}")
        print(f"     keys: {list(data.keys())}")
        if "tokens" in data:
            print(f"     tokens[:5]: {data['tokens'][:5]}")
        if "weights" in data:
            print(f"     weights[:5]: {[round(w,3) for w in data['weights'][:5]]}")
        break
if not found:
    print("     [WARN] No explain endpoint found or all errored")

print(f"\n{SEP}")
print("  API Tests Complete")
print(SEP)
