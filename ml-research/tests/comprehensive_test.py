"""
Comprehensive model diagnostic test.
Tests aspect identification, confidence, mixed sentiment, and XAI (attention).
Run from project root: python tests/comprehensive_test.py
"""

import sys, os, json
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from inference import SentimentPredictor

CKPT = os.path.join(project_root, "results", "cosmetic_sentiment_v1", "best_model.pt")

REVIEWS = [
    ("strongly_pos",
     "This lipstick is absolutely amazing! The colour is gorgeous, it lasts all day, smells divine and the price is very affordable."),
    ("strongly_neg",
     "Worst product ever. The colour is terrible, it fades in one hour, smells horrible like chemicals, and the packaging broke on arrival. Way overpriced."),
    ("mixed_colour_pos_smell_neg",
     "The colour is beautiful but the smell is absolutely awful. It also fades too quickly and the price is too high."),
    ("smell_neg_only",
     "Great colour and the texture is silky smooth, but the scent is absolutely disgusting and unbearable."),
    ("colour_pos_only",
     "The colour is breathtaking – such a rich vibrant shade. Perfect pigmentation."),
    ("neutral_generic",
     "It is an okay product. Nothing special but it does what it says on the box."),
    ("mixed_price_pos_texture_neg",
     "Very affordable price and great value for money. However the texture feels rough and heavy on the skin."),
    ("packing_neg",
     "Packaging was completely destroyed when it arrived. The box was crushed and the product was unusable."),
]

EXPECTED = {
    "strongly_pos":              {"colour": "positive", "smell": "positive", "price": "positive"},
    "strongly_neg":              {"colour": "negative", "smell": "negative", "price": "negative"},
    "mixed_colour_pos_smell_neg":{"colour": "positive", "smell": "negative", "price": "negative"},
    "smell_neg_only":            {"colour": "positive", "smell": "negative"},
    "colour_pos_only":           {"colour": "positive"},
    "neutral_generic":           {},  # no strong signal
    "mixed_price_pos_texture_neg":{"price": "positive", "texture": "negative"},
    "packing_neg":               {"packing": "negative"},
}

def run():
    print("=" * 70)
    print("  COMPREHENSIVE MODEL DIAGNOSTIC TEST")
    print("=" * 70)
    print(f"Checkpoint: {CKPT}")
    print()

    try:
        p = SentimentPredictor(CKPT)
    except Exception as e:
        print(f"[FATAL] Could not load model: {e}")
        import traceback; traceback.print_exc()
        return

    print(f"Aspects: {', '.join(p.aspect_names)}")

    total_expected = 0
    total_correct = 0
    all_confidences = []
    low_confidence_count = 0

    report = {}

    for rev_name, text in REVIEWS:
        expected_map = EXPECTED.get(rev_name, {})
        print(f"\n--- [{rev_name}] ---")
        print(f"  {text[:100]}")

        rev_results = {}
        for asp in p.aspect_names:
            r = p.predict(text, asp)
            probs = r["probabilities"]
            conf = r["confidence"]
            sentiment = r["sentiment"]

            all_confidences.append(conf)
            if conf < 0.4:
                low_confidence_count += 1

            expected_sentiment = expected_map.get(asp, None)
            if expected_sentiment is not None:
                total_expected += 1
                correct = (sentiment == expected_sentiment)
                if correct:
                    total_correct += 1
                marker = "OK " if correct else "ERR"
            else:
                marker = "   "

            rev_results[asp] = {
                "pred": sentiment, "conf": conf,
                "neg": probs["negative"], "neu": probs["neutral"], "pos": probs["positive"],
                "expected": expected_sentiment
            }

            flag = ""
            if expected_sentiment and sentiment != expected_sentiment:
                flag = f"  <-- WRONG (expected {expected_sentiment})"

            print(f"    [{marker}] {asp:<16} -> {sentiment:<8} conf={conf:.3f}"
                  f"  neg={probs['negative']:.2f} neu={probs['neutral']:.2f} pos={probs['positive']:.2f}{flag}")

        report[rev_name] = rev_results

    # --- Statistics ---
    avg_conf = sum(all_confidences) / len(all_confidences) if all_confidences else 0
    print("\n" + "=" * 70)
    print("  SUMMARY STATISTICS")
    print("=" * 70)
    print(f"  Aspect-level accuracy:   {total_correct}/{total_expected}  ({100*total_correct/max(total_expected,1):.1f}%)")
    print(f"  Average confidence:      {avg_conf:.3f}")
    print(f"  Low conf predictions (<0.4): {low_confidence_count}/{len(all_confidences)}")

    # --- Mixed sentiment check ---
    print("\n  [Mixed Sentiment Checks]")
    mixed_text = "The colour is beautiful but the smell is absolutely awful."
    mixed_preds = {}
    for asp in p.aspect_names:
        r = p.predict(mixed_text, asp)
        mixed_preds[asp] = (r["sentiment"], r["confidence"])
    colour_pred = mixed_preds.get("colour", ("?", 0))
    smell_pred  = mixed_preds.get("smell",  ("?", 0))
    print(f"  '{mixed_text}'")
    print(f"    colour -> {colour_pred[0]} (conf={colour_pred[1]:.3f})")
    print(f"    smell  -> {smell_pred[0]}  (conf={smell_pred[1]:.3f})")
    mixed_ok = (colour_pred[0] == "positive" and smell_pred[0] == "negative")
    print(f"    Mixed detection: {'PASS - colour=pos, smell=neg correctly detected' if mixed_ok else 'FAIL - cannot distinguish mixed sentiments'}")

    # --- XAI Attention Check ---
    print("\n  [Attention-Based XAI Check]")
    xai_text  = "The colour is beautiful but the smell is absolutely awful."
    xai_asp   = "colour"
    try:
        r = p.predict(xai_text, xai_asp, return_attention=True)
        if "attention" in r:
            tokens  = r["attention"]["tokens"]
            weights = r["attention"]["weights"]
            top5 = sorted(zip(tokens, weights), key=lambda x: x[1], reverse=True)[:5]
            print(f"  Top-5 attention tokens for '{xai_asp}' aspect:")
            for tok, wt in top5:
                print(f"    {tok:<20} {wt:.4f}")
        else:
            print("  [WARN] Attention not returned. return_attention may be broken.")
    except Exception as e:
        print(f"  [ERROR] XAI attention failed: {e}")

    # --- 3→4 class mapping preview ---
    print("\n  [3→4 Class Mapping Preview]")
    adapter_path = os.path.join(project_root, "website", "ml_models", "trained_model_adapter.py")
    import importlib.util
    spec = importlib.util.spec_from_file_location("adapter", adapter_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = mod._map_probs_3class_to_4class
    test_cases = [
        (0.8, 0.1, 0.1, "should be very_negative"),
        (0.1, 0.1, 0.8, "should be very_positive"),
        (0.33, 0.34, 0.33, "should be neutral-ish (ambiguous)"),
        (0.45, 0.3, 0.25, "leaning negative"),
    ]
    print(f"  {'neg':>6} {'neu':>6} {'pos':>6}  -> 4-class probs (vn / n / p / vp)  label")
    for neg, neu, pos, note in test_cases:
        p4 = fn(neg, neu, pos)
        import numpy as np
        label = mod.LABEL_NAMES_4CLASS[int(np.argmax(p4))]
        print(f"  {neg:.2f}  {neu:.2f}  {pos:.2f}  -> [{p4[0]:.2f}, {p4[1]:.2f}, {p4[2]:.2f}, {p4[3]:.2f}]  {label}   # {note}")

    print("\n" + "=" * 70)
    print("  Diagnostic complete.")
    print("=" * 70)

    # Save JSON report
    out_path = os.path.join(project_root, "tests", "comprehensive_diagnostic.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Full report saved to: {out_path}")

if __name__ == "__main__":
    run()
