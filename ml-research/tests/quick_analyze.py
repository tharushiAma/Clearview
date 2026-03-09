import sys, os

# tests/ is a subdirectory — add the project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
from inference import SentimentPredictor

p = SentimentPredictor('results/cosmetic_sentiment_v1/best_model.pt')
text = "Lipstick color is amazing, I don't like the smell and the price is bit high"

print("Review:", text)
print()
print("{:<16} {:<12} {:<8} {:<8} {:<8}".format("Aspect", "Prediction", "neg%", "neu%", "pos%"))
print("-" * 60)

expected = {
    "colour": "positive",
    "smell":  "negative",
    "price":  "negative",
}

all_correct = 0
for asp in p.aspect_names:
    r = p.predict(text, asp)
    probs = r['probabilities']
    exp = expected.get(asp, "?")
    match = " <-- WRONG (expected {})".format(exp) if exp != "?" and r['sentiment'] != exp else (" <-- correct" if exp != "?" else "")
    if exp != "?" and r['sentiment'] == exp:
        all_correct += 1
    print("{:<16} {:<12} {:.1f}%   {:.1f}%   {:.1f}%{}".format(
        asp, r['sentiment'],
        probs['negative']*100, probs['neutral']*100, probs['positive']*100,
        match))

print()
print("Key aspects correct: {}/3 (colour=pos, smell=neg, price=neg)".format(all_correct))
