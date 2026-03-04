"""
test_model_components.py
Unit-style tests for current model architecture components.
Does NOT require a trained checkpoint — tests forward-pass shapes and logic only.

Run from ml-research directory:
    python tests/test_model_components.py
"""

import sys, os
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR      = os.path.join(PROJECT_ROOT, "src")
UTILS_DIR    = os.path.join(PROJECT_ROOT, "utils")

for p in [PROJECT_ROOT, SRC_DIR, UTILS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import yaml
from models.model import AspectAwareRoBERTa, MultiAspectSentimentModel, create_model
from models.losses import FocalLoss, HybridLoss, AspectSpecificLossManager


def sep(title=""):
    print(f"\n{'='*60}")
    if title:
        print(f"  {title}")
        print(f"{'='*60}")

def ok(msg):  print(f"  [OK ] {msg}")
def err(msg): print(f"  [ERR] {msg}")
def skip(msg):print(f"  [SKIP] {msg}")


def load_config():
    cfg_path = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ── Test 1: AspectAwareRoBERTa forward pass ───────────────────────────────────
def test_aspect_aware_roberta(config):
    sep("Test 1: AspectAwareRoBERTa forward pass (all flag combinations)")
    model_cfg = config["model"]
    B, S = 2, 64

    for use_attention in [True, False]:
        for use_shared in [True, False]:
            label = f"attention={use_attention}, shared_clf={use_shared}"
            try:
                model = AspectAwareRoBERTa(
                    roberta_model=model_cfg["roberta_model"],
                    num_aspects=model_cfg["num_aspects"],
                    num_classes=model_cfg["num_classes"],
                    hidden_dim=model_cfg["hidden_dim"],
                    dropout=0.0,
                    use_aspect_attention=use_attention,
                    use_shared_classifier=use_shared,
                )
                model.eval()
                ids   = torch.randint(0, 1000, (B, S))
                mask  = torch.ones(B, S, dtype=torch.long)
                asp   = torch.zeros(B, dtype=torch.long)

                with torch.no_grad():
                    preds, attn_w, asp_repr = model(ids, mask, asp)

                assert preds.shape   == (B, model_cfg["num_classes"]), f"Wrong pred shape: {preds.shape}"
                assert attn_w.shape  == (B, S),                       f"Wrong attn shape: {attn_w.shape}"
                assert asp_repr.shape == (B, model_cfg["hidden_dim"]),  f"Wrong repr shape: {asp_repr.shape}"
                ok(label)
            except Exception as exc:
                err(f"{label}: {exc}")


# ── Test 2: MultiAspectSentimentModel (GCN on/off) ───────────────────────────
def test_full_model(config):
    sep("Test 2: MultiAspectSentimentModel (GCN on and off)")
    B, S = 2, 64
    ids  = torch.randint(0, 1000, (B, S))
    mask = torch.ones(B, S, dtype=torch.long)
    asp  = torch.zeros(B, dtype=torch.long)

    # Without GCN
    try:
        cfg_no_gcn = {**config}
        cfg_no_gcn["model"] = {**config["model"], "use_dependency_gcn": False}
        model = create_model(cfg_no_gcn)
        model.eval()
        with torch.no_grad():
            out = model(ids, mask, asp)
        assert out.shape == (B, config["model"]["num_classes"])
        ok(f"No-GCN forward pass — output shape {out.shape}")
    except Exception as exc:
        err(f"No-GCN: {exc}")

    # With GCN (dummy edges)
    try:
        model = create_model(config)
        model.eval()
        edge_index = [
            torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
            torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        ]
        with torch.no_grad():
            out = model(ids, mask, asp, edge_index=edge_index)
        assert out.shape == (B, config["model"]["num_classes"])
        ok(f"With-GCN forward pass — output shape {out.shape}")
    except Exception as exc:
        err(f"With-GCN: {exc}")

    # Return-attention mode
    try:
        model = create_model(config)
        model.eval()
        with torch.no_grad():
            out = model(ids, mask, asp, return_attention=True)
        assert isinstance(out, tuple) and len(out) == 4
        ok("return_attention=True returns 4-tuple (preds, attn, repr, gcn_out)")
    except Exception as exc:
        err(f"return_attention: {exc}")


# ── Test 3: Loss functions ────────────────────────────────────────────────────
def test_loss_functions(config):
    sep("Test 3: Loss functions")
    B = 8
    logits = torch.randn(B, 3)
    labels = torch.randint(0, 3, (B,))

    # FocalLoss
    try:
        fl = FocalLoss(gamma=2.0, alpha=torch.tensor([1.0, 1.5, 1.2]))
        loss = fl(logits, labels)
        assert loss.shape == (), f"Expected scalar, got {loss.shape}"
        assert not torch.isnan(loss), "FocalLoss is NaN"
        ok(f"FocalLoss: {loss.item():.4f}")
    except Exception as exc:
        err(f"FocalLoss: {exc}")

    # HybridLoss
    try:
        hl = HybridLoss(
            samples_per_class=[100, 50, 200],
            focal_gamma=2.0,
            cb_beta=0.999,
            weights={'focal': 1.0, 'cb': 0.5, 'dice': 0.3}
        )
        loss = hl(logits, labels)
        assert not torch.isnan(loss), "HybridLoss is NaN"
        ok(f"HybridLoss: {loss.item():.4f}")
    except Exception as exc:
        err(f"HybridLoss: {exc}")

    # AspectSpecificLossManager
    try:
        class_counts = config.get("training", {}).get("class_counts", {
            "stayingpower": [727, 244, 1076],
            "texture":      [639, 420, 2563],
            "smell":        [335, 274, 1668],
        })
        mgr = AspectSpecificLossManager(class_counts, config["training"])
        aspect_ids = torch.zeros(B, dtype=torch.long)
        aspects    = config["aspects"]["names"]
        loss, details = mgr.compute_loss(logits, labels, aspect_ids, aspects)
        assert not torch.isnan(loss), "AspectSpecificLossManager loss is NaN"
        ok(f"AspectSpecificLossManager: {loss.item():.4f}")
    except Exception as exc:
        err(f"AspectSpecificLossManager: {exc}")


# ── Test 4: Ablation flags do not break forward pass ─────────────────────────
def test_ablation_modes(config):
    sep("Test 4: Ablation flag correctness")
    B, S = 2, 32
    ids  = torch.randint(0, 1000, (B, S))
    mask = torch.ones(B, S, dtype=torch.long)
    asp  = torch.zeros(B, dtype=torch.long)

    ablation_cases = [
        {"use_aspect_attention": False, "use_dependency_gcn": True,  "use_shared_classifier": False},  # A2
        {"use_aspect_attention": True,  "use_dependency_gcn": False, "use_shared_classifier": False},  # A1
        {"use_aspect_attention": True,  "use_dependency_gcn": True,  "use_shared_classifier": True},   # A5
        {"use_aspect_attention": False, "use_dependency_gcn": False, "use_shared_classifier": True},   # combined
    ]
    for flags in ablation_cases:
        label = "  ".join(f"{k}={v}" for k, v in flags.items())
        try:
            cfg = {**config, "model": {**config["model"], **flags}}
            m   = create_model(cfg)
            m.eval()
            with torch.no_grad():
                out = m(ids, mask, asp)
            assert out.shape == (B, config["model"]["num_classes"])
            ok(label)
        except Exception as exc:
            err(f"{label}: {exc}")


# ── Test 5: Parameter count sanity ───────────────────────────────────────────
def test_parameter_count(config):
    sep("Test 5: Parameter count")
    try:
        model = create_model(config)
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        ok(f"Trainable parameters: {total:,}")
        # RoBERTa-base has ~125M params, total with heads should be > 100M
        assert total > 50_000_000, f"Suspiciously low param count: {total:,}"
        ok("Parameter count sanity check passed")
    except Exception as exc:
        err(f"Parameter count: {exc}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  CLEARVIEW — MODEL COMPONENT UNIT TESTS")
    print("=" * 60)
    print("  (No checkpoint required — tests forward-pass logic only)\n")

    try:
        config = load_config()
        ok(f"Config loaded from configs/config.yaml")
    except Exception as exc:
        err(f"Could not load config: {exc}")
        return

    test_aspect_aware_roberta(config)
    test_full_model(config)
    test_loss_functions(config)
    test_ablation_modes(config)
    test_parameter_count(config)

    print("\n" + "=" * 60)
    print("  Component tests complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
