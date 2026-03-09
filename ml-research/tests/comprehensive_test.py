"""
comprehensive_test.py
Comprehensive diagnostic test for the ClearView Multi-Aspect Sentiment Model.
Tests: prediction correctness, confidence, mixed sentiment detection,
       attention XAI, and optional Integrated Gradients / MSR delta.

Run from the ml-research directory:
    python tests/comprehensive_test.py --checkpoint outputs/cosmetic_sentiment_v1/best_model.pt

    # Also test XAI:
    python tests/comprehensive_test.py --checkpoint outputs/cosmetic_sentiment_v1/best_model.pt --xai

    # Save XAI charts:
    python tests/comprehensive_test.py --checkpoint outputs/cosmetic_sentiment_v1/best_model.pt --xai --save-charts
"""

import sys, os, json, argparse, time
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INFERENCE_DIR = os.path.join(PROJECT_ROOT, "outputs", "cosmetic_sentiment_v1", "evaluation")
SRC_DIR       = os.path.join(PROJECT_ROOT, "src")

for p in [PROJECT_ROOT, INFERENCE_DIR, SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from inference import SentimentPredictor   # noqa: E402  (path added above)

# ── Test cases ────────────────────────────────────────────────────────────────
REVIEWS = [
    ("strongly_pos",
     "This lipstick is absolutely amazing! The colour is gorgeous, it lasts all day, "
     "smells divine and the price is very affordable."),
    ("strongly_neg",
     "Worst product ever. The colour is terrible, it fades in one hour, smells horrible "
     "like chemicals, and the packaging broke on arrival. Way overpriced."),
    ("mixed_colour_pos_smell_neg",
     "The colour is beautiful but the smell is absolutely awful. "
     "It also fades too quickly and the price is too high."),
    ("smell_neg_texture_pos",
     "Great colour and the texture is silky smooth, "
     "but the scent is absolutely disgusting and unbearable."),
    ("colour_pos_only",
     "The colour is breathtaking – such a rich vibrant shade. Perfect pigmentation."),
    ("neutral_generic",
     "It is an okay product. Nothing special but it does what it says on the box."),
    ("mixed_price_pos_texture_neg",
     "Very affordable price and great value for money. "
     "However the texture feels rough and heavy on the skin."),
    ("packing_neg",
     "Packaging was completely destroyed when it arrived. "
     "The box was crushed and the product was unusable."),
    ("shipping_neg",
     "The product arrived two weeks late and was clearly mis-handled during delivery."),
    ("long_review",
     "Using this moisturizer for three months now. Texture is silky smooth and absorbs "
     "well. Skin feels noticeably softer. Staying power is excellent throughout the day. "
     "However the scent is very strong, almost medicinal — quite unpleasant. Packaging "
     "cracked when dropped. Very expensive for what you get."),
]

# Expected sentiment for key (review, aspect) pairs.
# Only aspects with a clear signal in the text are listed.
EXPECTED = {
    "strongly_pos":              {"colour": "positive", "smell": "positive", "price": "positive"},
    "strongly_neg":              {"colour": "negative", "smell": "negative", "price": "negative"},
    "mixed_colour_pos_smell_neg":{"colour": "positive", "smell": "negative"},
    "smell_neg_texture_pos":     {"texture": "positive", "smell": "negative"},
    "colour_pos_only":           {"colour": "positive"},
    "mixed_price_pos_texture_neg":{"price": "positive", "texture": "negative"},
    "packing_neg":               {"packing": "negative"},
    "shipping_neg":              {"shipping": "negative"},
}

MIXED_SENTIMENT_CASES = [
    {
        "text":   "The colour is beautiful but the smell is absolutely awful.",
        "checks": [("colour", "positive"), ("smell", "negative")],
        "label":  "colour=pos, smell=neg",
    },
    {
        "text":   "Great texture but way overpriced. The packaging is decent.",
        "checks": [("texture", "positive"), ("price", "negative")],
        "label":  "texture=pos, price=neg",
    },
]


def run(checkpoint: str, run_xai: bool = False, save_charts: bool = False):
    print("=" * 72)
    print("  CLEARVIEW MODEL — COMPREHENSIVE DIAGNOSTIC TEST")
    print("=" * 72)
    print(f"Checkpoint : {checkpoint}")
    print(f"XAI tests  : {'enabled' if run_xai else 'disabled (use --xai to enable)'}")
    print()

    # ── Load model ────────────────────────────────────────────────────────────
    try:
        predictor = SentimentPredictor(checkpoint)
    except FileNotFoundError:
        print(f"[FATAL] Checkpoint not found: {checkpoint}")
        print("        Train the model first: python src/models/train.py")
        return
    except Exception as exc:
        import traceback
        print(f"[FATAL] Could not load model: {exc}")
        traceback.print_exc()
        return

    aspects = predictor.aspect_names
    print(f"Aspects    : {', '.join(aspects)}")
    print(f"Device     : {predictor.device}")
    print()

    # ── Per-review prediction ─────────────────────────────────────────────────
    total_expected = total_correct = 0
    all_confidences = []
    low_conf_count  = 0
    report = {}

    for rev_name, text in REVIEWS:
        expected_map = EXPECTED.get(rev_name, {})
        print(f"\n[{rev_name}]")
        print(f"  \"{text[:110]}{'…' if len(text) > 110 else ''}\"")

        rev_results = {}
        t0 = time.time()
        for asp in aspects:
            r         = predictor.predict(text, asp)
            probs     = r["probabilities"]
            conf      = r["confidence"]
            sentiment = r["sentiment"]

            all_confidences.append(conf)
            if conf < 0.40:
                low_conf_count += 1

            expected = expected_map.get(asp)
            if expected is not None:
                total_expected += 1
                correct = (sentiment == expected)
                total_correct += (1 if correct else 0)
                marker = "OK " if correct else "ERR"
                flag   = "" if correct else f"  <-- expected {expected}"
            else:
                marker = "   "
                flag   = ""

            rev_results[asp] = {
                "pred": sentiment, "conf": conf,
                "neg":  probs["negative"], "neu": probs["neutral"], "pos": probs["positive"],
                "expected": expected,
            }
            print(f"  [{marker}] {asp:<15} -> {sentiment:<9} conf={conf:.3f}"
                  f"  neg={probs['negative']:.2f} neu={probs['neutral']:.2f} "
                  f"pos={probs['positive']:.2f}{flag}")

        elapsed = (time.time() - t0) * 1000
        print(f"  Inference time: {elapsed:.0f} ms for {len(aspects)} aspects")
        report[rev_name] = rev_results

    # ── Summary statistics ────────────────────────────────────────────────────
    avg_conf = float(np.mean(all_confidences)) if all_confidences else 0
    acc_pct  = 100 * total_correct / max(total_expected, 1)

    print("\n" + "=" * 72)
    print("  SUMMARY STATISTICS")
    print("=" * 72)
    print(f"  Aspect-level accuracy   : {total_correct}/{total_expected}  ({acc_pct:.1f}%)")
    print(f"  Average confidence      : {avg_conf:.3f}")
    print(f"  Low-conf predictions    : {low_conf_count}/{len(all_confidences)} "
          f"(conf < 0.40)")

    # ── Mixed Sentiment Detection ─────────────────────────────────────────────
    print("\n  [Mixed Sentiment Resolution Checks]")
    mixed_pass = 0
    for case in MIXED_SENTIMENT_CASES:
        checks_pass = True
        results_line = []
        for asp, expected_sent in case["checks"]:
            r = predictor.predict(case["text"], asp)
            ok = (r["sentiment"] == expected_sent)
            checks_pass = checks_pass and ok
            results_line.append(f"{asp}={r['sentiment']}(conf={r['confidence']:.2f})")

        status = "PASS" if checks_pass else "FAIL"
        mixed_pass += (1 if checks_pass else 0)
        print(f"  [{status}] {case['label']}")
        print(f"        Text  : \"{case['text'][:70]}\"")
        print(f"        Preds : {', '.join(results_line)}")

    print(f"\n  Mixed sentiment: {mixed_pass}/{len(MIXED_SENTIMENT_CASES)} cases correctly resolved")

    # ── Attention XAI Check ───────────────────────────────────────────────────
    print("\n  [Attention-Based XAI Check]")
    xai_text = "The colour is beautiful but the smell is absolutely awful."
    for xai_asp in ["colour", "smell"]:
        try:
            r = predictor.predict(xai_text, xai_asp, return_attention=True)
            if "attention" not in r:
                print(f"  [WARN] attention not returned for '{xai_asp}'")
                continue
            tokens  = r["attention"]["tokens"]
            weights = r["attention"]["weights"]
            top5    = sorted(zip(tokens, weights), key=lambda x: x[1], reverse=True)[:5]
            print(f"  Top-5 attention tokens — aspect '{xai_asp}' (pred={r['sentiment']}):")
            for tok, wt in top5:
                bar = "█" * int(wt * 40)
                print(f"    {tok:<20} {bar} {wt:.4f}")
        except Exception as exc:
            print(f"  [ERROR] Attention XAI failed for '{xai_asp}': {exc}")

    # ── Optional: Integrated Gradients + MSR Delta ────────────────────────────
    if run_xai:
        print("\n  [Integrated Gradients Check]")
        ig_text = "The colour is amazing but the smell is quite off-putting."
        try:
            charts_dir = os.path.join(PROJECT_ROOT, "tests", "xai_charts")
            os.makedirs(charts_dir, exist_ok=True)
            ig_save = os.path.join(charts_dir, "ig_colour.png") if save_charts else None
            predictor.explain_with_integrated_gradients(
                ig_text, "colour", n_steps=30, top_k=10, save_path=ig_save
            )
            print("  [OK] Integrated Gradients completed")
        except ImportError:
            print("  [SKIP] captum not installed. Run: pip install captum")
        except Exception as exc:
            print(f"  [ERROR] Integrated Gradients failed: {exc}")

        print("\n  [MSR Delta Check]")
        msr_text = "I love the colour of this lipstick but the smell is absolutely revolting."
        try:
            msr_save = os.path.join(charts_dir, "msr_colour.png") if save_charts else None
            predictor.explain_msr_delta(msr_text, "colour", top_k=8, save_path=msr_save)
            print("  [OK] MSR Delta completed")
        except Exception as exc:
            print(f"  [ERROR] MSR Delta failed: {exc}")

    # ── Final verdict ─────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    if acc_pct >= 70 and mixed_pass == len(MIXED_SENTIMENT_CASES):
        print("  [RESULT] PASS — Model performing correctly")
    elif acc_pct >= 50:
        print("  [RESULT] PARTIAL — Model needs more training (accuracy low or mixed detection failing)")
    else:
        print("  [RESULT] FAIL — Model not ready; low accuracy or mixed sentiment not resolved")
    print("=" * 72)

    # Save JSON report
    out_path = os.path.join(PROJECT_ROOT, "tests", "comprehensive_diagnostic.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Full report saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ClearView Model Diagnostic Test")
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(PROJECT_ROOT, "outputs", "cosmetic_sentiment_v1", "best_model.pt"),
        help="Path to the trained model checkpoint",
    )
    parser.add_argument(
        "--xai", action="store_true",
        help="Run Integrated Gradients and MSR Delta XAI tests (requires captum)",
    )
    parser.add_argument(
        "--save-charts", action="store_true",
        help="Save XAI chart PNGs to tests/xai_charts/",
    )
    args = parser.parse_args()
    run(args.checkpoint, run_xai=args.xai, save_charts=args.save_charts)
