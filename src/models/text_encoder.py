"""
Text Encoder Module for MFGF TVPR
Uses BERT to extract global semantic representation [CLS] token: f_Text.
"""

from typing import Optional
import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    """
    Text Encoder extracting global semantic representation from BERT [CLS] token.
    Maps to embed_dim (default 768).
    """
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        embed_dim: int = 768,
        pretrained: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.has_hf_bert = False

        try:
            from transformers import AutoModel
            if pretrained:
                self.bert = AutoModel.from_pretrained(model_name)
            else:
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(model_name)
                self.bert = AutoModel.from_config(config)
            self.has_hf_bert = True
            bert_hidden_dim = self.bert.config.hidden_size
        except Exception:
            # Fallback lightweight Transformer Encoder for offline or isolated testing
            self.bert = None
            bert_hidden_dim = 768
            self.vocab_size = 30522
            self.word_emb = nn.Embedding(self.vocab_size, bert_hidden_dim)
            self.pos_emb = nn.Parameter(torch.randn(1, 128, bert_hidden_dim) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=bert_hidden_dim, nhead=8, dim_feedforward=2048, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, bert_hidden_dim))

        if bert_hidden_dim != embed_dim:
            self.proj = nn.Linear(bert_hidden_dim, embed_dim)
        else:
            self.proj = nn.Identity()

        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input_ids: Tensor of shape [B, L]
            attention_mask: Tensor of shape [B, L]
        Returns:
            f_Text: Global semantic feature tensor of shape [B, embed_dim]
        """
        if self.has_hf_bert:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            # Use [CLS] token representation (token at index 0)
            cls_feature = outputs.last_hidden_state[:, 0, :] # [B, hidden_dim]
        else:
            # Fallback transformer forward
            b, l = input_ids.shape
            x = self.word_emb(input_ids) # [B, L, D]
            cls_tokens = self.cls_token.expand(b, -1, -1) # [B, 1, D]
            x = torch.cat((cls_tokens, x), dim=1) # [B, L+1, D]
            seq_len = x.size(1)
            x = x + self.pos_emb[:, :seq_len, :]
            
            src_key_padding_mask = None
            if attention_mask is not None:
                # Add 1 for CLS token
                cls_mask = torch.ones(b, 1, device=attention_mask.device, dtype=attention_mask.dtype)
                combined_mask = torch.cat((cls_mask, attention_mask), dim=1)
                src_key_padding_mask = (combined_mask == 0)

            x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
            cls_feature = x[:, 0, :] # [B, D]

        f_Text = self.proj(cls_feature)
        f_Text = self.layer_norm(f_Text)
        return f_Text
