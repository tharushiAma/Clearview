import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel
from typing import List, Dict, Optional


class AspectAwareAttention(nn.Module):
    def __init__(self, hidden_size: int, num_aspects: int):
        super().__init__()
        self.num_aspects = num_aspects
        self.aspect_queries = nn.Parameter(torch.randn(num_aspects, hidden_size))

        self.attention = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4, dropout=0.1, batch_first=True)
            for _ in range(num_aspects)
        ])
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, sequence_output: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        B, T, H = sequence_output.shape
        aspect_reprs: List[torch.Tensor] = []
        attn_weights: List[torch.Tensor] = []

        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None

        for i in range(self.num_aspects):
            query = self.aspect_queries[i].unsqueeze(0).unsqueeze(1).expand(B, 1, -1)  # [B,1,H]
            repr_i, attn_i = self.attention[i](
                query=query,
                key=sequence_output,
                value=sequence_output,
                key_padding_mask=key_padding_mask,
                need_weights=True
            )
            repr_i = self.layer_norm(repr_i.squeeze(1))  # [B,H]
            attn_i = attn_i.squeeze(1)                   # [B,T]
            aspect_reprs.append(repr_i)
            attn_weights.append(attn_i)

        return aspect_reprs, attn_weights


class CrossAspectInteraction(nn.Module):
    def __init__(self, hidden_size: int, num_aspects: int):
        super().__init__()
        self.num_aspects = num_aspects
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4, dropout=0.1, batch_first=False)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
        )
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, aspect_reprs: List[torch.Tensor]):
        x = torch.stack(aspect_reprs, dim=0)  # [A,B,H]
        attn_out, attn_weights = self.cross_attn(x, x, x, need_weights=True)
        x = self.ln1(x + attn_out)
        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)
        refined = [x[i] for i in range(self.num_aspects)]
        return refined, attn_weights


class ConflictDetector(nn.Module):
    """
    FIXED: conflict features = probs + entropy + sentiment contrast
    """
    def __init__(self, num_aspects: int, num_classes: int = 4):
        super().__init__()
        self.num_aspects = num_aspects
        in_dim = num_aspects * num_classes + num_aspects + 1

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, aspect_logits: List[torch.Tensor]) -> torch.Tensor:
        probs = [F.softmax(lg, dim=-1) for lg in aspect_logits]  # [B,C]
        probs_cat = torch.cat(probs, dim=-1)                    # [B,A*C]

        entropies = []
        argmaxes = []
        for p in probs:
            ent = -torch.sum(p * torch.log(p + 1e-8), dim=-1, keepdim=True)  # [B,1]
            entropies.append(ent)
            argmaxes.append(torch.argmax(p, dim=-1, keepdim=True).float())   # [B,1]

        entropies = torch.cat(entropies, dim=-1)  # [B,A]
        argmaxes = torch.cat(argmaxes, dim=-1)    # [B,A]
        contrast = (argmaxes.max(dim=1)[0] - argmaxes.min(dim=1)[0]).unsqueeze(1)  # [B,1]

        feat = torch.cat([probs_cat, entropies, contrast], dim=-1)
        return torch.sigmoid(self.mlp(feat))  # [B,1]


class MaskedFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, ignore_index: int = -100):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, alpha: Optional[torch.Tensor] = None) -> torch.Tensor:
        valid = targets != self.ignore_index
        if valid.sum().item() == 0:
            return torch.tensor(0.0, device=logits.device)

        logits_v = logits[valid]
        targets_v = targets[valid]

        ce = F.cross_entropy(logits_v, targets_v, reduction="none", weight=alpha)
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


class ImprovedRoBERTaHierarchical(nn.Module):
    def __init__(
        self,
        num_aspects: int = 7,
        num_classes: int = 4,
        aspect_names: Optional[List[str]] = None,
        hidden_dropout: float = 0.3,
        gamma_values: Optional[Dict[str, float]] = None,
        msr_strength: float = 0.3,
        roberta_name: str = "roberta-base",
    ):
        super().__init__()
        self.num_aspects = num_aspects
        self.num_classes = num_classes
        self.aspect_names = aspect_names or [
            'stayingpower', 'texture', 'smell', 'price', 'colour', 'shipping', 'packing'
        ]

        self.roberta = RobertaModel.from_pretrained(roberta_name)
        hidden_size = self.roberta.config.hidden_size

        self.aspect_attention = AspectAwareAttention(hidden_size, num_aspects)
        self.cross_aspect = CrossAspectInteraction(hidden_size, num_aspects)

        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(hidden_dropout),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Dropout(hidden_dropout),
                nn.Linear(hidden_size // 2, num_classes)
            )
            for _ in range(num_aspects)
        ])

        self.conflict_detector = ConflictDetector(num_aspects, num_classes)

        self.msr_strength = msr_strength
        self.msr_refiners = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(hidden_dropout),
                nn.Linear(hidden_size + 1, hidden_size // 2),
                nn.GELU(),
                nn.Dropout(hidden_dropout),
                nn.Linear(hidden_size // 2, num_classes)
            )
            for _ in range(num_aspects)
        ])

        if gamma_values is None:
            gamma_values = {asp: 2.0 for asp in self.aspect_names}
            gamma_values["price"] = 3.0
            gamma_values["smell"] = 2.5
            gamma_values["packing"] = 2.5

        self.aspect_losses = nn.ModuleDict({
            asp: MaskedFocalLoss(gamma=gamma_values.get(asp, 2.0))
            for asp in self.aspect_names
        })

    def forward(self, input_ids, attention_mask, enable_msr: bool = True, return_attention: bool = False) -> Dict:
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        seq = out.last_hidden_state  # [B,T,H]

        aspect_features, aspect_attn = self.aspect_attention(seq, attention_mask)
        refined_features, cross_attn = self.cross_aspect(aspect_features)

        base_logits = [self.classifiers[i](refined_features[i]) for i in range(self.num_aspects)]
        conflict_score = self.conflict_detector(base_logits)  # [B,1]

        if not enable_msr or self.msr_strength <= 0:
            res = {"aspect_logits": base_logits, "conflict_score": conflict_score}
            if return_attention:
                res["aspect_attention"] = aspect_attn
                res["cross_attention"] = cross_attn
            return res

        # FIXED: gated refinement
        gate = conflict_score.detach()  # [B,1]
        refined_logits = []
        for i in range(self.num_aspects):
            x = torch.cat([refined_features[i], conflict_score], dim=1)  # [B,H+1]
            delta = self.msr_refiners[i](x)
            refined_logits.append(base_logits[i] + self.msr_strength * gate * delta)

        # Option B/C: Does MSR resolve the conflict?
        conflict_refined = self.conflict_detector(refined_logits)
        
        res = {
            "aspect_logits": refined_logits, 
            "conflict_score": conflict_refined, # Official score becomes refined
            "conflict_base": conflict_score      # Keep base for research logging
        }
        if return_attention:
            res["aspect_attention"] = aspect_attn
            res["cross_attention"] = cross_attn
        return res

    def compute_loss(self, forward_output: Dict, labels: torch.Tensor, conflict_labels=None,
                     loss_weights={"aspect": 1.0, "conflict": 0.5},
                     class_weights: Optional[torch.Tensor] = None):
        aspect_logits = forward_output["aspect_logits"]
        conflict_score = forward_output["conflict_score"].squeeze(-1)

        aspect_loss = 0.0
        valid_aspects = 0
        for i, asp in enumerate(self.aspect_names):
            loss_i = self.aspect_losses[asp](aspect_logits[i], labels[:, i], alpha=class_weights)
            if loss_i.item() > 0:
                aspect_loss += loss_i
                valid_aspects += 1
        aspect_loss = aspect_loss / max(valid_aspects, 1)

        conflict_loss = torch.tensor(0.0, device=labels.device)
        if conflict_labels is not None:
            conflict_loss = F.binary_cross_entropy(conflict_score, conflict_labels.float())

        total_loss = loss_weights["aspect"] * aspect_loss + loss_weights["conflict"] * conflict_loss
        return total_loss, {"total": float(total_loss.item()), "aspect": float(aspect_loss.item()),
                            "conflict": float(conflict_loss.item())}

    @torch.no_grad()
    def predict(self, input_ids, attention_mask, enable_msr: bool = True, return_base_conflict: bool = False):
        self.eval()
        out = self.forward(input_ids, attention_mask, enable_msr=enable_msr)
        preds = []
        probs = []
        for lg in out["aspect_logits"]:
            p = F.softmax(lg, dim=-1)
            preds.append(torch.argmax(p, dim=-1))
            probs.append(p)
        preds = torch.stack(preds, dim=1)  # [B,A]
        probs = torch.stack(probs, dim=1)  # [B,A,C]
        
        conflict = out["conflict_score"].squeeze(-1)  # [B]
        if return_base_conflict and "conflict_base" in out:
            base_conflict = out["conflict_base"].squeeze(-1)
            return preds, probs, conflict, base_conflict
            
        return preds, probs, conflict
