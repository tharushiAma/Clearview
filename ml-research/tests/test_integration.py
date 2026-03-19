"""
test_integration.py
Integration test for the full website-to-model pipeline.
Validates that the TrainedModelAdapter correctly wraps inference.py,
produces correct output format, and passes all aspect predictions through.

Run from the ml-research directory:
    python tests/test_integration.py

    # With a specific checkpoint:
    python tests/test_integration.py --checkpoint outputs/cosmetic_sentiment_v1/best_model.pt
"""

import sys, os, time, argparse

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INFERENCE_DIR = os.path.join(PROJECT_ROOT, "outputs", "cosmetic_sentiment_v1", "evaluation")
SRC_DIR       = os.path.join(PROJECT_ROOT, "src")

for p in [PROJECT_ROOT, INFERENCE_DIR, SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import importlib.util

def load_adapter_module():
    # Adapter now lives in ml-research/inference_bridge/ (moved from website/ml_models/)
    adapter_path = os.path.join(PROJECT_ROOT, "inference_bridge", "trained_model_adapter.py")
    if not os.path.exists(adapter_path):
        return None, f"Adapter not found: {adapter_path}"
    spec = importlib.util.spec_from_file_location("trained_model_adapter", adapter_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, None


# ── Test cases ────────────────────────────────────────────────────────────────
#  These reviews cover: short, long, mixed, strongly positive, strongly negative,
#  and edge cases (single word, emoji-heavy).
REVIEWS = [
    ("positive_clear",  "This foundation is amazing! Color perfection, stays all day, smells divine."),
    ("negative_clear",  "Worst lipstick ever. Fades in an hour, chemical stench, packaging snapped."),
    ("mixed_col_sml",   "Love the color and texture. But the smell is absolutely horrible and fades fast."),
    ("mixed_pkg_price", "Great packaging and reasonable price. But skin feels weird and color is patchy."),
    ("neutral_bland",   "It is an okay product. Nothing special but does what it says."),
    ("short_great",     "Great!"),
    ("short_terrible",  "Terrible."),
    ("long_detailed",   (
        "Using this moisturizer for three months. Texture is silky smooth, absorbs well. "
        "Skin feels noticeably softer. Staying power is excellent throughout the day. "
        "However the scent is very strong, almost medicinal — quite unpleasant. "
        "Packaging cracked when dropped. Very expensive for what you get."
    )),
    ("shipping_late",   "Product arrived two weeks late. Was clearly mis-handled."),
    ("all_aspects",     (
        "The colour is beautiful, texture is silky, stays all day, smells amazing, "
        "well-priced, delivered quickly and the packaging is premium quality."
    )),
]


# ── Required output format ────────────────────────────────────────────────────
# Based on what the website frontend expects from TrainedModelAdapter.predict()
VALID_SENTIMENTS   = {"negative", "neutral", "positive", "not_mentioned"}
REQUIRED_PRED_KEYS = {"name", "confidence", "sentiment"}   # minimum required


def check_adapter_output(result: dict, review_name: str, expected_aspects: list) -> list:
    """Validate the adapter output format."""
    errors = []

    # Top-level key (adapter returns {'predictions': [...], ...})
    if "predictions" not in result and "aspects" not in result:
        errors.append(f"{review_name}: Missing 'predictions' or 'aspects' key")
        return errors

    predictions = result.get("predictions") or result.get("aspects") or []

    if len(predictions) != len(expected_aspects):
        errors.append(
            f"{review_name}: Expected {len(expected_aspects)} predictions, got {len(predictions)}"
        )

    returned_aspects = {p.get("name") or p.get("aspect") for p in predictions}
    for asp in expected_aspects:
        if asp not in returned_aspects:
            errors.append(f"{review_name}: Missing aspect '{asp}' in predictions")

    for pred in predictions:
        asp_name = pred.get("name") or pred.get("aspect", "?")

        sentiment = pred.get("sentiment") or pred.get("label") or pred.get("predicted_class")
        if sentiment not in VALID_SENTIMENTS:
            errors.append(
                f"{review_name} | {asp_name}: Invalid sentiment '{sentiment}'"
            )

        conf = pred.get("confidence", -1)
        if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
            errors.append(
                f"{review_name} | {asp_name}: Confidence out of range: {conf}"
            )

        probs = pred.get("probs") or pred.get("probabilities")
        if probs is not None:
            if isinstance(probs, dict):
                total = sum(probs.values())
            else:
                total = sum(probs)
            if not (0.97 <= total <= 1.03):
                errors.append(
                    f"{review_name} | {asp_name}: Probs don't sum to 1 (sum={total:.4f})"
                )

    return errors


# ── Test runner ───────────────────────────────────────────────────────────────
def run_tests(checkpoint: str):
    SEP = "=" * 72
    print(SEP)
    print("  CLEARVIEW — FULL INTEGRATION TEST: Website Adapter")
    print(SEP)
    print(f"Checkpoint : {checkpoint}")
    print(f"Reviews    : {len(REVIEWS)}")
    print()

    # Try loading the website adapter
    adapter_mod, err = load_adapter_module()
    if adapter_mod is None:
        print(f"[WARN] {err}")
        print("       Falling back to direct SentimentPredictor test.")
        run_direct_predictor_test(checkpoint)
        return

    try:
        adapter = adapter_mod.TrainedModelAdapter(checkpoint)
        aspect_names = adapter.aspect_names
        print(f"[OK] Adapter loaded on: {adapter.device}")
        print(f"     Aspects: {', '.join(aspect_names)}")
        print()
    except Exception as exc:
        import traceback
        print(f"[FATAL] Adapter load failed: {exc}")
        traceback.print_exc()
        return

    all_errors, successful = [], 0
    print(f"{'Review':<18} {'Time(ms)':>10}  {'Status':<10} Predictions summary")
    print("-" * 72)

    for name, text in REVIEWS:
        try:
            t0     = time.time()
            result = adapter.predict(text)
            elapsed_ms = (time.time() - t0) * 1000

            errors = check_adapter_output(result, name, aspect_names)
            all_errors.extend(errors)

            preds = result.get("predictions") or result.get("aspects") or []
            summary = "  ".join(
                f"{(p.get('name') or p.get('aspect', '?'))[:5]}/{(p.get('sentiment') or p.get('label', '?'))[:3]}"
                for p in preds[:4]
            )
            status = "[OK]" if not errors else "[ERR]"
            print(f"{name:<18} {elapsed_ms:>8.0f}ms  {status:<10} {summary}")
            if not errors:
                successful += 1

        except Exception as exc:
            print(f"{name:<18}  [PREDICT_ERR] {exc}")
            all_errors.append(f"{name}: exception: {exc}")

    print()
    print(SEP)
    print("  INTEGRATION TEST SUMMARY")
    print(SEP)
    print(f"  Successful reviews : {successful}/{len(REVIEWS)}")
    print(f"  Format errors      : {len(all_errors)}")

    if all_errors:
        print("\n  Errors:")
        for e in all_errors:
            print(f"    - {e}")

    print()
    if successful == len(REVIEWS) and not all_errors:
        print(f"[RESULT] PASS — All {len(REVIEWS)} reviews processed correctly.")
    else:
        print(f"[RESULT] FAIL — {len(all_errors)} errors, {len(REVIEWS)-successful} failed reviews.")


def run_direct_predictor_test(checkpoint: str):
    """Fallback: test SentimentPredictor directly (without website adapter)."""
    from inference import SentimentPredictor

    SEP = "=" * 72
    print("\n[Fallback] Direct SentimentPredictor test")
    print(SEP)

    try:
        predictor = SentimentPredictor(checkpoint)
    except Exception as exc:
        print(f"[FATAL] Could not load predictor: {exc}")
        return

    all_ok, total = 0, 0
    for name, text in REVIEWS:
        try:
            t0 = time.time()
            results = predictor.predict_all_aspects(text)
            elapsed_ms = (time.time() - t0) * 1000
            total += 1

            aspects_out = list(results.keys())
            summary = "  ".join(f"{a[:5]}/{r['sentiment'][:3]}" for a, r in list(results.items())[:4])
            print(f"{name:<20} {elapsed_ms:>7.0f}ms  [OK] {summary}")
            all_ok += 1
        except Exception as exc:
            print(f"{name:<20}  [ERR] {exc}")
            total += 1

    print(f"\nResult: {all_ok}/{total} reviews passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full Integration Test for ClearView")
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(PROJECT_ROOT, "outputs", "cosmetic_sentiment_v1", "best_model.pt"),
        help="Path to the trained model checkpoint",
    )
    args = parser.parse_args()
    run_tests(args.checkpoint)
