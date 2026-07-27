from typing import List

import torch
import torch.nn as nn
from einops import rearrange
from torch import FloatTensor, LongTensor

from tamer.datamodule import vocab
from tamer.model.pos_enc import WordPosEnc
from tamer.model.transformer.arm import AttentionRefinementModule
from tamer.model.transformer.transformer_decoder import (
    TransformerDecoder,
    TransformerDecoderLayer,
)
from tamer.utils.generation_utils import DecodeModel
from tamer.model.adapter import GatedBottleneckAdapter


class LBR(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        return self.relu(x)


class StructSimOneDir(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.trm = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, nhead=8, dim_feedforward=1024, dropout=0.3
            ),
            num_layers=1,
        )
        self.to_q = nn.Linear(d_model, d_model)
        self.to_k = nn.Linear(d_model, d_model)
        self.to_sim = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(d_model, 1))

    def forward(self, tgt, tgt_key_padding_mask):
        tgt = self.trm(
            src=tgt, src_key_padding_mask=tgt_key_padding_mask
        )
        q = self.to_q(tgt)
        k = self.to_k(tgt)
        q = rearrange(q, "t b d -> b t () d")
        k = rearrange(k, "l b d -> b () l d")
        sim = self.to_sim(q + k).squeeze(-1)
        sim = sim.masked_fill(tgt_key_padding_mask[:, None, :], float("-inf"))
        return sim, tgt


class StructSim(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.l2r_struct_sim = StructSimOneDir(d_model)
        self.r2l_struct_sim = StructSimOneDir(d_model)

    def forward(self, out, src_key_padding_mask):
        l2r_out, r2l_out = torch.chunk(out, 2, dim=1)
        l2r_kp_mask, r2l_kp_mask = torch.chunk(src_key_padding_mask, 2, dim=0)
        
        l2r_sim, l2r_feature = self.l2r_struct_sim(l2r_out, l2r_kp_mask)
        r2l_sim, r2l_feature = self.r2l_struct_sim(r2l_out, r2l_kp_mask)

        sim = torch.cat((l2r_sim, r2l_sim), dim=0)
        feature = torch.cat((l2r_feature, r2l_feature), dim=1)
        return sim, feature


class FusionModule(nn.Module):
    """Gate explicit decoder features with Tree-Aware semantic features."""

    def __init__(self, d_model: int):
        super().__init__()
        self.w_att = nn.Linear(2 * d_model, d_model)

    def forward(self, explicit: FloatTensor, tree_aware: FloatTensor) -> FloatTensor:
        attention = torch.sigmoid(self.w_att(torch.cat((explicit, tree_aware), dim=2)))
        return attention * tree_aware + (1.0 - attention) * explicit
    

def _build_transformer_decoder(
    d_model: int,
    nhead: int,
    num_decoder_layers: int,
    dim_feedforward: int,
    dropout: float,
    dc: int,
    cross_coverage: bool,
    self_coverage: bool,
) -> nn.TransformerDecoder:
    decoder_layer = TransformerDecoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )
    if cross_coverage or self_coverage:
        arm = AttentionRefinementModule(
            nhead, dc, cross_coverage, self_coverage)
    else:
        arm = None

    decoder = TransformerDecoder(decoder_layer, num_decoder_layers, arm)
    return decoder


class Decoder(DecodeModel):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        dc: int,
        cross_coverage: bool,
        self_coverage: bool,
        vocab_size: int = 114,
        use_fusion: bool = False,
        use_decoder_adapter: bool = False,
        adapter_bottleneck_dim: int = 64,
        adapter_dropout: float = 0.1,
        adapter_gate_init_bias: float = -2.0,
    ):
        super().__init__()

        self.word_embed = nn.Sequential(
            nn.Embedding(vocab_size, d_model), nn.LayerNorm(d_model)
        )

        self.pos_enc = WordPosEnc(d_model=d_model)

        self.norm = nn.LayerNorm(d_model)

        self.model = _build_transformer_decoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
        )
        self.struct_sim = StructSim(d_model)
        self.decoder_adapter = (
            GatedBottleneckAdapter(
                d_model=d_model,
                bottleneck_dim=adapter_bottleneck_dim,
                dropout=adapter_dropout,
                gate_init_bias=adapter_gate_init_bias,
            )
            if use_decoder_adapter
            else None
        )
        self.use_fusion = use_fusion
        if use_fusion:
            # These names and dimensions match the released version_3 state_dict.
            self.fusion = FusionModule(d_model)
            self.fusion_proj = nn.Linear(d_model, vocab_size)
            self.exp_proj = nn.Linear(d_model, vocab_size)
        else:
            self.proj = nn.Linear(d_model, vocab_size)


    def _build_attention_mask(self, length):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.full(
            (length, length), fill_value=1, dtype=torch.bool, device=self.device
        )
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(
        self, src: FloatTensor, src_mask: LongTensor, tgt: LongTensor
    ) -> FloatTensor:
        """generate output for tgt

        Parameters
        ----------
        src : FloatTensor
            [b, h, w, d]
        src_mask: LongTensor
            [b, h, w]
        tgt : LongTensor
            [b, l]

        Returns
        -------
        FloatTensor
            [b, l, vocab_size]
        """
        _, l = tgt.size()
        tgt_mask = self._build_attention_mask(l)
        tgt_pad_mask = tgt == vocab.PAD_IDX

        tgt = self.word_embed(tgt)  # [b, l, d]
        tgt = self.pos_enc(tgt)  # [b, l, d]
        tgt = self.norm(tgt)

        h = src.shape[1]
        src = rearrange(src, "b h w d -> (h w) b d")
        src_mask = rearrange(src_mask, "b h w -> b (h w)")
        tgt = rearrange(tgt, "b l d -> l b d")

        out = self.model(
            tgt=tgt,
            memory=src,
            height=h,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_mask,
        )
        if self.decoder_adapter is not None:
            out = self.decoder_adapter(out)

        sim, tree_aware = self.struct_sim(out, tgt_pad_mask)
        explicit = rearrange(out, "l b d -> b l d")
        if self.use_fusion:
            tree_aware = rearrange(tree_aware, "l b d -> b l d")
            fused = self.fusion(explicit, tree_aware)
            return self.exp_proj(explicit), self.fusion_proj(fused), sim
        return self.proj(explicit), sim

    def transform(
        self, src: List[FloatTensor], src_mask: List[LongTensor], input_ids: LongTensor
    ) -> FloatTensor:
        assert len(src) == 1 and len(src_mask) == 1
        outputs = self(src[0], src_mask[0], input_ids)
        if self.use_fusion:
            _, fusion_logits, sim = outputs
            return fusion_logits, sim
        return outputs
