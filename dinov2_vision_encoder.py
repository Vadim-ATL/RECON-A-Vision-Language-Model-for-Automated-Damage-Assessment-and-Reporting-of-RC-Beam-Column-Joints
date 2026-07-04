import torch
import torch.nn as nn
from transformers import AutoModel


# ---------------------------------------------------------------------------
# Expert: cross-attention block (field token queries patch features)
# ---------------------------------------------------------------------------

class Expert(nn.Module):
    def __init__(self, d: int = 768, heads: int = 8):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.ff    = nn.Sequential(
            nn.Linear(d, d * 2),
            nn.GELU(),
            nn.Linear(d * 2, d),
        )
        self.norm2 = nn.LayerNorm(d)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        o, _ = self.attn(q, kv, kv)
        q = self.norm1(q + o)
        q = self.norm2(q + self.ff(q))
        return q


# ---------------------------------------------------------------------------
# Field-Aware MoE Vision Encoder
# ---------------------------------------------------------------------------

class FieldAwareMoE(nn.Module):
    """
    DINOv2 backbone + per-field MoE routing.

    Key fixes vs previous version:
    1. temperature = 1.0  (was 0.5 → near one-hot routing → 3/4 experts dead)
    2. field_proj: each field projects patches differently before expert routing
       so experts can actually specialise per field
    3. dtype cast at entry (float32) prevents autocast fp16 crash in DINOv2 conv
    4. Small field_token init kept (was already correct at ×0.02)
    5. FIX (NEW): backbone forward no longer wrapped in a blanket torch.no_grad().
       Previously freeze_backbone=True unfroze the last 4 transformer blocks
       (requires_grad=True) but the forward pass ran entirely inside
       torch.no_grad(), which silently disables autograd for EVERY parameter
       regardless of requires_grad -- the unfreezing was a no-op. We now run
       the backbone forward with grad enabled, and instead rely on
       requires_grad alone (already set correctly above) to control which
       blocks actually receive gradients. The frozen blocks still get no
       gradient because their parameters have requires_grad=False; the
       activations flowing through them just won't be needlessly retained
       for backward in those frozen layers since no leaf there needs a grad.
    """

    def __init__(self, fields_list: list, n_experts: int,
                 d: int = 768, freeze_backbone: bool = True):
        super().__init__()

        self.backbone = AutoModel.from_pretrained("facebook/dinov2-base")

        # Freeze everything first, then explicitly unfreeze the last 4 blocks.
        # (freeze_backbone flag kept for API compatibility; with the no_grad
        # fix above, requires_grad is now the ONLY thing controlling which
        # blocks train, so this flag should stay True unless you intend to
        # fine-tune the entire backbone.)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.fields_list = fields_list
        self.n_experts   = n_experts
        n = len(fields_list)

        for block in self.backbone.encoder.layer[-4:]:
            for p in block.parameters():
                p.requires_grad = True
        n_trainable = sum(p.requires_grad for p in self.backbone.parameters())
        print(f"[FieldAwareMoE] Last 4 DINOv2 blocks unfrozen "
              f"({n_trainable} trainable parameter tensors in backbone).")

        # Small init — consistent with QFormer query init fix
        self.field_tokens = nn.Parameter(torch.randn(1, n, d) * 0.02)

        self.experts = nn.ModuleList([Expert(d) for _ in range(n_experts)])
        self.routers = nn.ModuleList([nn.Linear(d, n_experts) for _ in range(n)])

        # FIX: per-field patch projection so each field sees a different
        # linear view of the patches → experts can specialise
        self.field_proj = nn.ModuleList([nn.Linear(d, d) for _ in range(n)])

        # FIX: temperature 1.0 (was 0.5 → collapsed routing)
        self.temperature = 1.0

    # ------------------------------------------------------------------
    def forward(self, pv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pv: [B, 3, 224, 224]  float32 pixel values
        Returns:
            [B, 256 + n_fields, 768]  — patches + field tokens concatenated
        """
        backbone_device = next(self.backbone.parameters()).device
        backbone_dtype  = next(self.backbone.parameters()).dtype  # float32

        # Cast to backbone dtype BEFORE autocast can introduce fp16
        pv = pv.to(device=backbone_device, dtype=backbone_dtype)

        B = pv.size(0)

        # ---- DINOv2 patch features ----------------------------------------
        # NOTE: no longer wrapped in torch.no_grad() -- see class docstring
        # fix #5. Frozen blocks (requires_grad=False) still contribute no
        # gradient; the last 4 unfrozen blocks now actually train.
        patches = self.backbone(pixel_values=pv).last_hidden_state[:, 1:]
        # patches: [B, 256, 768]

        # ---- MoE field routing -------------------------------------------
        tokens = self.field_tokens.expand(B, -1, -1)   # [B, n_fields, 768]
        field_feats = []

        for i in range(len(self.fields_list)):
            q = tokens[:, i:i+1, :]                    # [B, 1, 768]

            # Field-specific patch projection (NEW)
            proj_patches = self.field_proj[i](patches)  # [B, 256, 768]

            # Routing: soft gate over experts
            gate_logits = self.routers[i](q.squeeze(1)) / self.temperature
            gates = torch.softmax(gate_logits, dim=-1)  # [B, n_experts]

            # Each expert cross-attends field token to projected patches
            outs = torch.stack(
                [exp(q, proj_patches)[:, 0, :] for exp in self.experts],
                dim=1,
            )  # [B, n_experts, 768]

            # Weighted sum of expert outputs
            feat = (outs * gates.unsqueeze(-1)).sum(dim=1)  # [B, 768]
            field_feats.append(feat)

        field_tokens_out = torch.stack(field_feats, dim=1)  # [B, n_fields, 768]

        # Concatenate patch features + field-aware tokens
        return torch.cat([patches, field_tokens_out], dim=1)  # [B, 261, 768]