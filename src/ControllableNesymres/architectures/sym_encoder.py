import torch
import torch.nn as nn
from .utils import float2bit

class SymEncoderWithAttentionDec(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.trg_pad_idx = cfg.architecture.trg_pad_idx

        self.tok_embeddings = nn.Embedding(num_embeddings=cfg.architecture.num_tokens_condition, embedding_dim=cfg.architecture.embedding_dim_condition, padding_idx=cfg.architecture.src_pad_idx)
        self.pos_embedding = nn.Embedding(num_embeddings=cfg.max_description_seq_len, embedding_dim=cfg.architecture.embedding_dim_condition)
        self.anchor_embedding = nn.Embedding(num_embeddings=cfg.max_description_seq_len, embedding_dim=cfg.architecture.embedding_dim_condition)

        self.trasf_enc = nn.TransformerDecoderLayer(d_model=cfg.architecture.embedding_dim_condition,
                                              nhead=cfg.architecture.num_heads,
                                              dim_feedforward=cfg.architecture.dec_pf_dim,
                                              dropout=cfg.architecture.dropout,
        )
        self.enc = nn.TransformerDecoder(self.trasf_enc, num_layers=cfg.architecture.cond_num_layers)

        self.cfg = cfg

        # Bit16
        self.bit16 = cfg.architecture.bit16
        self.norm = cfg.architecture.norm
        self.step_up_layer = nn.Linear(16, cfg.architecture.embedding_dim_condition)

    
    def make_src_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).float()
        trg_pad_mask = (
            trg_pad_mask.masked_fill(trg_pad_mask == 0, float("-inf"))
            .masked_fill(trg_pad_mask == 1, float(0.0))
            .type_as(trg)
        )
        return trg_pad_mask


    def forward(self, batch, device=None):
        symbolic_conditioning = batch['symbolic_conditioning'].long()
        numerical_conditioning = batch['numerical_conditioning'].float()
        array = torch.cat([symbolic_conditioning, numerical_conditioning], dim=1)
        mask = self.make_src_mask(array)

        te = self.tok_embeddings(symbolic_conditioning)

        #(numerical_conditioning == np.inf).any(axis=1)[64]
        numerical_conditioning = numerical_conditioning.unsqueeze(-1)

        if numerical_conditioning.shape[1] > 0:
            if self.bit16:
                x = float2bit(numerical_conditioning, device=device)
                x = x.view(x.shape[0],x.shape[1],-1)
                if self.norm:
                    x = (x-0.5)*2

                numerical_te = self.step_up_layer(x)
                te = torch.cat([te, numerical_te], dim=1)
            else:
                raise NotImplementedError("Only bit16 is implemented for now")
        
        

        pos = torch.arange(0, te.shape[1], device=device).unsqueeze(0).repeat(te.shape[0], 1)
        pos_emb = self.pos_embedding(pos)
        encoder_input = te + pos_emb
        anch = torch.arange(0, self.cfg.architecture.num_features, device=device).unsqueeze(0).repeat(te.shape[0], 1)
        memory = self.anchor_embedding(anch)

        enc_embedding = self.enc(memory.permute(1,0,2),encoder_input.permute(1,0,2),memory_key_padding_mask=mask.bool())
        
        enc_embedding = enc_embedding.permute(1,0,2)

        if torch.isnan(enc_embedding).any():
            breakpoint()

        return enc_embedding