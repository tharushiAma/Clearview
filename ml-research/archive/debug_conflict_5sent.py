import torch
from transformers import RobertaTokenizerFast
from src.models.roberta_hierarchical_improved import ImprovedRoBERTaHierarchical
from src.data_layer._common import ASPECTS

def debug_conflict():
    ckpt = "outputs/smoke_run/best_model.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = ImprovedRoBERTaHierarchical(num_aspects=len(ASPECTS), num_classes=4).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    
    sentences = [
        "The texture is nice.", # Clear Positive
        "The price is too high.", # Clear Negative
        "The texture is nice but the price is too high.", # Mixed (High Conflict)
        "It is okay.", # Neutral
        "I like the color and the smell is great." # Multi-Positive (Lower Conflict than Mixed)
    ]
    
    print(f"{'Text':<50} | {'Base Conf':<10} | {'Refined Conf':<10} | {'Delta':<10}")
    print("-" * 90)
    
    for text in sentences:
        enc = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            _, _, conf_refined, conf_base = model.predict(
                enc['input_ids'], 
                enc['attention_mask'], 
                enable_msr=True, 
                return_base_conflict=True
            )
        
        base = conf_base.item()
        refined = conf_refined.item()
        print(f"{text[:48]:<50} | {base:<10.4f} | {refined:<10.4f} | {refined-base:<10.4f}")

if __name__ == "__main__":
    debug_conflict()
