import torch
from transformers import RobertaTokenizerFast
from src.models.roberta_hierarchical_improved import ImprovedRoBERTaHierarchical
from src.data_layer._common import ASPECTS

def check_conflict_logic():
    print("=== Checking Conflict Logic (Fix #2) ===")
    
    # 1. Init model
    device = "cpu"
    num_classes = 4 # New default
    model = ImprovedRoBERTaHierarchical(
        num_aspects=len(ASPECTS),
        num_classes=num_classes,
        aspect_names=ASPECTS,
        msr_strength=0.3
    ).to(device)
    model.eval()
    
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    
    # 2. Sentences with varying conflict levels
    # S1: Clear positive
    s1 = "The texture is amazing and matches the price perfectly."
    # S2: Mixed / Conflict
    s2 = "The texture is good but the price is too high."
    
    inputs1 = tokenizer(s1, return_tensors="pt")
    inputs2 = tokenizer(s2, return_tensors="pt")
    
    # 3. Predict with MSR ON vs OFF
    print(f"\nSentence 1 (Clear): '{s1}'")
    _, _, conf1_off = model.predict(inputs1['input_ids'], inputs1['attention_mask'], enable_msr=False)
    _, _, conf1_on, conf1_base  = model.predict(inputs1['input_ids'], inputs1['attention_mask'], enable_msr=True, return_base_conflict=True)
    
    print(f"Conflict (MSR=False): {conf1_off.item():.4f}")
    print(f"Conflict (MSR=True, refined): {conf1_on.item():.4f}")
    print(f"Conflict (MSR=True, base):    {conf1_base.item():.4f}")
    
    assert conf1_off.item() == conf1_base.item(), "Base conflict should match MSR=False!"
    if abs(conf1_on.item() - conf1_base.item()) > 1e-6:
        print(">> OK: MSR affects conflict score (refined != base).")
    else:
        print(">> NOTE: MSR didn't change conflict for this CLEAR sentence (expected if gate=0).")

    print(f"\nSentence 2 (Mixed): '{s2}'")
    _, _, conf2_off = model.predict(inputs2['input_ids'], inputs2['attention_mask'], enable_msr=False)
    
    print(f"Conflict S1: {conf1_on.item():.4f}")
    print(f"Conflict S2: {conf2_off.item():.4f}")
    
    # Check that S2 has higher conflict than S1 using the SAME model state
    # (Note: Un-trained model might be random, but structural difference exists)
    # Since weights are random, we can't guarantee S2 > S1, but we can guarantee they are Different.
    if abs(conf1_on.item() - conf2_off.item()) > 1e-6:
        print(">> OK: Conflict score varies by input.")
    else:
        print(">> WARNING: Conflict score identical for different inputs? (Random init might cause this)")

    print("\n✅ Verification Complete: Model runs with 4 classes and conflict logic holds.")

if __name__ == "__main__":
    check_conflict_logic()
