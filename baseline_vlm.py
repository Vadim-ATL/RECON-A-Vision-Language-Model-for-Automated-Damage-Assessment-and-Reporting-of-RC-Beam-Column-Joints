import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from qformer_bridge import QFormer, LLMProjector


class BaselineVLM(nn.Module):
    def __init__(self, use_lora=False):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vision_encoder = None
        self.qformer   = QFormer().to(self.device)
        self.projector = LLMProjector().to(self.device)

        print("Tokenizer: meta-llama/Llama-3.2-3B-Instruct...")
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token    = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        print("[Loading LLM Backbone: meta-llama/Llama-3.2-3B-Instruct...")
        self.language_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-3B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="sdpa",
        )

        print("LLM Backbone is Frozen")
        self.language_model.eval()
        for p in self.language_model.parameters():
            p.requires_grad = False

    def reform_visual_tokens(self, pixel_values: torch.Tensor, labels_meta=None):
        feats = self.vision_encoder(pixel_values.to(self.device))
        
        # New Q-former returns projected features AND aux losses
        qout, aux_losses = self.qformer(feats, labels_meta)
        qout = qout.to(next(self.projector.parameters()).device)
        
        tokens = self.projector(qout)
        tokens = tokens.to(dtype=self.language_model.dtype)
        return tokens, aux_losses                                    

    def forward(self, pixel_values, input_ids, labels=None, attention_mask=None, labels_meta=None):
        visual_tokens, aux_losses = self.reform_visual_tokens(pixel_values, labels_meta)
        
        # ✅ CORRECT position — right after reform_visual_tokens
        vis_var_loss = -visual_tokens.var(dim=1).mean()

        text_embeds = self.language_model.get_input_embeddings()(input_ids.to(self.device))
        bos_embeds  = text_embeds[:, :1, :]
        rest_embeds = text_embeds[:, 1:, :]
        inputs_embeds = torch.cat([bos_embeds, visual_tokens, rest_embeds], dim=1)

        full_labels = None
        if labels is not None:
            vis_labels  = torch.full((visual_tokens.size(0), visual_tokens.size(1)), -100, device=self.device)
            full_labels = torch.cat([labels[:, :1], vis_labels, labels[:, 1:]], dim=1)

        full_mask = None
        if attention_mask is not None:
            vis_mask  = torch.ones((visual_tokens.size(0), visual_tokens.size(1)), device=self.device)
            full_mask = torch.cat([attention_mask[:, :1], vis_mask, attention_mask[:, 1:]], dim=1)

        lm_out = self.language_model(
            inputs_embeds=inputs_embeds,
            labels=full_labels,
            attention_mask=full_mask,
            return_dict=True,
        )

        return lm_out, aux_losses
    
    @torch.no_grad()
    def generate(self, pixel_values, input_ids, max_new_tokens=256, **kwargs):
        # Must match the forward logic!
        visual_tokens, _ = self.reform_visual_tokens(pixel_values)
        text_embeds = self.language_model.get_input_embeddings()(input_ids.to(self.device))
        
        bos_embeds = text_embeds[:, :1, :]
        rest_embeds = text_embeds[:, 1:, :]
        inputs_embeds = torch.cat([bos_embeds, visual_tokens, rest_embeds], dim=1)

        vis_mask = torch.ones((visual_tokens.size(0), visual_tokens.size(1)), device=self.device)
        full_mask = torch.cat([torch.ones((input_ids.shape[0], 1), device=self.device), vis_mask, torch.ones((input_ids.shape[0], input_ids.shape[1]-1), device=self.device)], dim=1)

        kwargs.pop("input_ids", None)
        return self.language_model.generate(inputs_embeds=inputs_embeds, attention_mask=full_mask, max_new_tokens=max_new_tokens, **kwargs)