#!/usr/bin/env python3
"""
Comprehensive test suite for ClearView UI Demo
Tests all API endpoints as the browser would use them
"""
import requests
import json
import time
from typing import Dict, Any

BASE_URL = "http://localhost:3000"

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_success(msg: str):
    print(f"{Colors.GREEN}✓ {msg}{Colors.END}")

def print_error(msg: str):
    print(f"{Colors.RED}✗ {msg}{Colors.END}")

def print_info(msg: str):
    print(f"{Colors.BLUE}ℹ {msg}{Colors.END}")

def print_warning(msg: str):
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.END}")

def test_predict_basic():
    """Test basic prediction functionality"""
    print_info("Testing /api/predict with basic input...")
    
    payload = {
        "text": "This phone is amazing but very expensive",
        "msr_enabled": True,
        "msr_strength": 0.3
    }
    
    try:
        r = requests.post(f"{BASE_URL}/api/predict", json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        
        # Validate response structure
        assert "aspects" in data, "Missing 'aspects' in response"
        assert "conflict_prob" in data, "Missing 'conflict_prob' in response"
        assert isinstance(data["aspects"], list), "aspects should be a list"
        assert len(data["aspects"]) > 0, "aspects list is empty"
        
        # Validate aspect structure
        aspect = data["aspects"][0]
        required_fields = ["name", "label", "confidence", "probs", "before", "after", "changed_by_msr"]
        for field in required_fields:
            assert field in aspect, f"Missing '{field}' in aspect"
        
        print_success(f"Basic prediction works. Got {len(data['aspects'])} aspects. Conflict prob: {data['conflict_prob']:.4f}")
        return True
    except Exception as e:
        print_error(f"Basic prediction failed: {e}")
        return False

def test_predict_multiple_sentences():
    """Test prediction with complex multi-sentence input"""
    print_info("Testing /api/predict with complex input...")
    
    payload = {
        "text": "I love the color and texture. The packing was terrible. The price is too high but quality is good.",
        "msr_enabled": True,
        "msr_strength": 0.3
    }
    
    try:
        r = requests.post(f"{BASE_URL}/api/predict", json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        
        # Check for MSR changes
        msr_changes = [a for a in data["aspects"] if a["changed_by_msr"]]
        if msr_changes:
            print_success(f"Complex prediction works. MSR modified {len(msr_changes)} aspect(s)")
        else:
            print_warning("Complex prediction works but MSR made no changes")
        return True
    except Exception as e:
        print_error(f"Complex prediction failed: {e}")
        return False

def test_predict_without_msr():
    """Test prediction with MSR disabled"""
    print_info("Testing /api/predict with MSR disabled...")
    
    payload = {
        "text": "Great product",
        "msr_enabled": False,
        "msr_strength": 0.0
    }
    
    try:
        r = requests.post(f"{BASE_URL}/api/predict", json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        
        # All aspects should show no MSR changes
        msr_changes = [a for a in data["aspects"] if a["changed_by_msr"]]
        assert len(msr_changes) == 0, "MSR should not make changes when disabled"
        
        print_success("Prediction without MSR works correctly")
        return True
    except Exception as e:
        print_error(f"Prediction without MSR failed: {e}")
        return False

def test_explain_conflict():
    """Test XAI explanation for conflict detection"""
    print_info("Testing /api/explain for conflict (this will take 30-60 seconds)...")
    
    payload = {
        "text": "Good quality but expensive",
        "aspect": "all",
        "methods": ["ig"],
        "msr_enabled": True,
        "msr_strength": 0.3
    }
    
    try:
        start = time.time()
        r = requests.post(f"{BASE_URL}/api/explain", json=payload, timeout=120)
        duration = time.time() - start
        r.raise_for_status()
        data = r.json()
        
        # Validate response structure
        assert "text" in data, "Missing 'text' in response"
        assert "ig_conflict" in data, "Missing 'ig_conflict' in response"
        assert "aspects" in data, "Missing 'aspects' in response"
        
        # Validate ig_conflict structure
        ig = data["ig_conflict"]
        assert "attributions" in ig, "Missing 'attributions' in ig_conflict"
        assert isinstance(ig["attributions"], list), "attributions should be a list"
        
        token_count = len(ig["attributions"])
        print_success(f"XAI conflict explanation works. {token_count} tokens analyzed in {duration:.1f}s")
        return True
    except Exception as e:
        print_error(f"XAI explanation failed: {e}")
        return False

def test_explain_single_aspect():
    """Test XAI explanation for a single aspect"""
    print_info("Testing /api/explain for single aspect...")
    
    payload = {
        "text": "Very expensive product",
        "aspect": "price",
        "methods": ["ig"],
        "msr_enabled": True,
        "msr_strength": 0.3
    }
    
    try:
        r = requests.post(f"{BASE_URL}/api/explain", json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        
        # Should have aspect-specific explanations
        assert "aspects" in data, "Missing 'aspects' in response"
        aspect_keys = list(data["aspects"].keys())
        
        print_success(f"Single aspect XAI works. Analyzed aspects: {aspect_keys}")
        return True
    except Exception as e:
        print_error(f"Single aspect XAI failed: {e}")
        return False

def test_metrics():
    """Test metrics endpoint"""
    print_info("Testing /api/metrics...")
    
    try:
        r = requests.get(f"{BASE_URL}/api/metrics", timeout=10)
        r.raise_for_status()
        data = r.json()
        
        # Validate response structure
        expected_keys = ["overall_macro_f1_4class", "overall_macro_f1_sentiment", "conflict", "msr_error_reduction"]
        for key in expected_keys:
            assert key in data, f"Missing '{key}' in metrics response"
        
        # Validate conflict nested structure
        conflict = data["conflict"]
        assert "conf_f1_macro" in conflict, "Missing 'conf_f1_macro' in conflict"
        assert "roc_auc" in conflict, "Missing 'roc_auc' in conflict"
        
        print_success(f"Metrics endpoint works. F1-4class: {data['overall_macro_f1_4class']:.4f}")
        return True
    except Exception as e:
        print_error(f"Metrics endpoint failed: {e}")
        return False

def test_logs():
    """Test logs endpoint"""
    print_info("Testing /api/logs...")
    
    try:
        r = requests.get(f"{BASE_URL}/api/logs", timeout=10)
        r.raise_for_status()
        data = r.json()
        
        # Should return an array (might be empty)
        assert isinstance(data, list), "Logs should be an array"
        
        if len(data) > 0:
            print_success(f"Logs endpoint works. Retrieved {len(data)} log entries")
        else:
            print_warning("Logs endpoint works but returned no entries (expected if no logs exist)")
        return True
    except Exception as e:
        print_error(f"Logs endpoint failed: {e}")
        return False

def test_frontend_accessible():
    """Test that frontend page loads"""
    print_info("Testing frontend page accessibility...")
    
    try:
        r = requests.get(BASE_URL, timeout=10)
        r.raise_for_status()
        
        # Check for React content
        html = r.text
        assert "ClearView" in html or "next" in html.lower(), "Frontend doesn't contain expected content"
        
        print_success("Frontend page loads successfully")
        return True
    except Exception as e:
        print_error(f"Frontend page failed to load: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("ClearView UI Demo - Comprehensive Test Suite")
    print("="*60 + "\n")
    
    results = {}
    
    # Test 1: Frontend
    results["Frontend Loads"] = test_frontend_accessible()
    print()
    
    # Test 2: Basic Prediction
    results["Basic Prediction"] = test_predict_basic()
    print()
    
    # Test 3: Complex Prediction
    results["Complex Prediction"] = test_predict_multiple_sentences()
    print()
    
    # Test 4: Prediction without MSR
    results["Prediction (MSR Off)"] = test_predict_without_msr()
    print()
    
    # Test 5: Metrics
    results["Metrics Display"] = test_metrics()
    print()
    
    # Test 6: Logs
    results["Logs Display"] = test_logs()
    print()
    
    # Test 7: XAI Conflict (slow)
    print_warning("Next test will take 30-60 seconds...")
    results["XAI Conflict Explanation"] = test_explain_conflict()
    print()
    
    # Test 8: XAI Single Aspect (slow)
    print_warning("Next test will take 20-40 seconds...")
    results["XAI Single Aspect"] = test_explain_single_aspect()
    print()
    
    # Summary
    print("="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = f"{Colors.GREEN}PASS{Colors.END}" if passed_test else f"{Colors.RED}FAIL{Colors.END}"
        print(f"{test_name:.<40} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print_success("\n🎉 ALL TESTS PASSED! Application is fully functional.")
        return 0
    else:
        print_error(f"\n❌ {total - passed} test(s) failed. Please review errors above.")
        return 1

if __name__ == "__main__":
    exit(main())
