import torch
import torch.nn as nn
from transformers import Blip2Model

# ---------------------------------------------------------------------------
# LLM Projector: 768 → 3072  (two-layer MLP)
# ---------------------------------------------------------------------------
class LLMProjector(nn.Module):
    """
    Maps Q-Former output (768) to LLaMA-3 hidden dim (3072).
    """
    def __init__(self, qformer_dim: int = 768, llm_dim: int = 3072):
        super().__init__()
        self.projector = nn.Sequential(
            nn.LayerNorm(qformer_dim),
            nn.Linear(qformer_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)   # [B, 32, 3072]

# ---------------------------------------------------------------------------
# Supervised Q-Former Bridge (Using Pretrained BLIP-2)
# ---------------------------------------------------------------------------
class QFormer(nn.Module):
    """
    Replaces the from-scratch Q-Former. Uses BLIP-2 pretrained weights 
    and adds task-specific semantic routing for the first 10 tokens.

    FIX (vs previous version): damage_head previously spanned 6 token slots
    (q_out[:, 4:10, :]) to match a 6-class DAMAGE_CLASSES list that included
    a "None" class. The training config's real taxonomy only has 5 damage
    classes (Flexural cracking, Diagonal shear cracking, Concrete crushing,
    Concrete spalling, Rebar exposure) -- "None" was dead capacity that
    never matched any real label. Slot range trimmed to q_out[:, 4:9, :]
    (5 tokens) so damage_head's logit count matches labels_meta['damage']
    exactly. This also frees up token slot 9, which previously did nothing
    useful (it was the head for the dead "None" class); it now simply isn't
    used for auxiliary supervision, but still flows into the LLM-facing
    32-token output unaffected.
    """
    def __init__(self, pretrained_model: str = "Salesforce/blip2-opt-2.7b"):
        super().__init__()
        print(f"[QFormer] Loading pretrained weights from {pretrained_model}...")
        
        # Load BLIP-2 to extract the robust, pre-trained Q-Former backbone
        blip2 = Blip2Model.from_pretrained(pretrained_model)
        
        self.qformer = blip2.qformer
        # Clone query tokens so they remain intact when we delete the base model
        self.query_tokens = nn.Parameter(blip2.query_tokens.clone())
        
        # Explicitly free up memory (critical for 1x RTX 5090)
        del blip2
        self.vision_proj = nn.Linear(768, 1408)
        # -------------------------------------------------------------------
        # Task-Specific Supervision Heads 
        # (Anchors the visual tokens to structural concepts)
        # -------------------------------------------------------------------
        
        # Token 0 -> Joint Type (4 classes)
        self.jtype_head = nn.Linear(768, 4)
        
        # Token 1 -> Failure Mode (3 classes: B, J, BJ)
        self.failure_head = nn.Linear(768, 4)
        
        # Token 2 -> Joint Severity (5 classes: DS0 to DS4)
        self.jsev_head = nn.Linear(768, 5)
        
        # Token 3 -> Beam Severity (5 classes: DS0 to DS4)
        self.bsev_head = nn.Linear(768, 5)
        
        # Tokens 4 through 8 -> Damage Types (5 classes, multi-label)
        # FIX: was tokens 4-9 (6 slots) for a 6-class list including an
        # unused "None" class. Now 5 slots for the real 5-class taxonomy.
        # Mapping each of the 5 tokens to 1 logit preserves a strict 1-to-1
        # token-to-damage-type semantic alignment.
        self.damage_head = nn.Linear(768, 1)

    def forward(self, image_features: torch.Tensor, labels_meta: dict = None):
        """
        Args:
            image_features: [B, 261, 768] from FieldAwareMoE
            labels_meta: Optional dictionary containing ground truth for auxiliary losses
                         e.g., {'jtype': tensor, 'failure': tensor, ...}
                         
        Returns:
            q_out: [B, 32, 768] Base Q-Former outputs to feed into LLMProjector
            aux_losses: Dictionary of calculated tracking losses (if labels_meta provided)
        """
        B = image_features.size(0)
        device = image_features.device
        image_features = self.vision_proj(image_features)
        # Expand the pretrained query tokens across the current batch size
        query_embeds = self.query_tokens.expand(B, -1, -1).to(device)
        
        # Create an attention mask for the input image patches (all patches are valid)
        encoder_attention_mask = torch.ones(
            B, image_features.size(1), 
            device=device, 
            dtype=torch.long
        )
        
        # Forward pass through the pre-trained Q-Former cross-attention layers
        qformer_outputs = self.qformer(
            query_embeds=query_embeds,
            encoder_hidden_states=image_features,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True
        )
        
        q_out = qformer_outputs.last_hidden_state  # [B, 32, 768]
        
        # -------------------------------------------------------------
        # Extract Semantic Logits
        # -------------------------------------------------------------
        logits = {
            'jtype': self.jtype_head(q_out[:, 0, :]),           # [B, 4]
            'failure': self.failure_head(q_out[:, 1, :]),       # [B, 3]
            'jsev': self.jsev_head(q_out[:, 2, :]),             # [B, 5]
            'bsev': self.bsev_head(q_out[:, 3, :]),             # [B, 5]
            # FIX: 4:9 (5 slots) instead of 4:10 (6 slots) -- squeeze dim=-1
            # gives [B, 5] logits matching the real 5-class damage taxonomy.
            'damage': self.damage_head(q_out[:, 4:9, :]).squeeze(-1)
        }

        aux_losses = {}
        
        # -------------------------------------------------------------
        # Calculate Auxiliary Losses (if labels are provided during training)
        # -------------------------------------------------------------
        if labels_meta is not None:
            ce_loss = nn.CrossEntropyLoss()
            bce_loss = nn.BCEWithLogitsLoss()
            
            # Single-label classification
            aux_losses['jtype'] = ce_loss(logits['jtype'], labels_meta['jtype'])
            aux_losses['failure'] = ce_loss(logits['failure'], labels_meta['failure'])
            aux_losses['jsev'] = ce_loss(logits['jsev'], labels_meta['jsev'])
            aux_losses['bsev'] = ce_loss(logits['bsev'], labels_meta['bsev'])
            
            # Multi-label classification (ensure labels_meta['damage'] is a float tensor)
            aux_losses['damage'] = bce_loss(logits['damage'], labels_meta['damage'].float())
            
            # Inverse Variance Loss: Ensures the 32 tokens don't collapse into identical vectors.
            # Adds 1e-6 to prevent division by zero.
            token_variance = q_out.var(dim=1).mean()
            aux_losses['var_loss'] = -token_variance

            
        return q_out, aux_losses