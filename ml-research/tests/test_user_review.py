"""Test the exact review the user is testing."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from inference import SentimentPredictor

ckpt = os.path.join("..", "results", "cosmetic_sentiment_v1", "best_model.pt")
p = SentimentPredictor(ckpt, temperature=0.5)

review = "Lipstick color is amazing, I don't like the smell and the price is bit high"
print(f"\nReview: {review}\n")
print(f"{'Aspect':14s} {'Predicted':10s} {'Confidence':12s} {'neg%':7s} {'neu%':7s} {'pos%':7s}")
print("-" * 60)

# Expected: colour=positive, smell=negative, price=negative
expected = {'colour': 'positive', 'smell': 'negative', 'price': 'negative'}

for asp in p.aspect_names:
    result = p.predict(review, asp)
    s = result['sentiment']
    c = result['confidence']
    neg = result['probabilities']['negative']
    neu = result['probabilities']['neutral']
    pos = result['probabilities']['positive']
    exp = expected.get(asp, '')
    flag = ''
    if exp:
        flag = '✓' if s == exp else f'✗ (expected {exp})'
    print(f"{asp:14s} {s:10s} {c*100:8.1f}%   {neg*100:5.1f}% {neu*100:5.1f}% {pos*100:5.1f}%  {flag}")
