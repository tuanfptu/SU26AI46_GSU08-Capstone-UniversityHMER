from typing import List

import pytorch_lightning as pl
import torch
from torch import FloatTensor, LongTensor

from tamer.utils.utils import Hypothesis

from .decoder import Decoder
from .encoder import Encoder
from .adapter import GatedBottleneckAdapter


class TAMER(pl.LightningModule):
    def __init__(
        self,
        d_model: int,
        growth_rate: int,
        num_layers: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        dc: int,
        cross_coverage: bool,
        self_coverage: bool,
        vocab_size: int = 114,
        use_fusion: bool = False,
        use_encoder_adapter: bool = False,
        use_decoder_adapter: bool = False,
        adapter_bottleneck_dim: int = 64,
        adapter_dropout: float = 0.1,
        adapter_gate_init_bias: float = -2.0,
    ):
        super().__init__()

        self.encoder = Encoder(
            d_model=d_model, growth_rate=growth_rate, num_layers=num_layers
        )
        self.encoder_adapter = (
            GatedBottleneckAdapter(
                d_model=d_model,
                bottleneck_dim=adapter_bottleneck_dim,
                dropout=adapter_dropout,
                gate_init_bias=adapter_gate_init_bias,
            )
            if use_encoder_adapter
            else None
        )
        self.decoder = Decoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
            vocab_size=vocab_size,
            use_fusion=use_fusion,
            use_decoder_adapter=use_decoder_adapter,
            adapter_bottleneck_dim=adapter_bottleneck_dim,
            adapter_dropout=adapter_dropout,
            adapter_gate_init_bias=adapter_gate_init_bias,
        )

    def forward(
        self, img: FloatTensor, img_mask: LongTensor, tgt: LongTensor
    ) -> FloatTensor:
        """run img and bi-tgt

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h, w]
        img_mask: LongTensor
            [b, h, w]
        tgt : LongTensor
            [2b, l]

        Returns
        -------
        FloatTensor
            [2b, l, vocab_size]
        """
        feature, mask = self.encoder(img, img_mask)  # [b, t, d]
        if self.encoder_adapter is not None:
            feature = self.encoder_adapter(feature)
        feature = torch.cat((feature, feature), dim=0)  # [2b, t, d]
        mask = torch.cat((mask, mask), dim=0)

        return self.decoder(feature, mask, tgt)

    def beam_search(
        self,
        img: FloatTensor,
        img_mask: LongTensor,
        beam_size: int,
        max_len: int,
        alpha: float,
        early_stopping: bool,
        temperature: float,
        **kwargs,
    ) -> List[Hypothesis]:
        """run bi-direction beam search for given img

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h', w']
        img_mask: LongTensor
            [b, h', w']
        beam_size : int
        max_len : int

        Returns
        -------
        List[Hypothesis]
        """
        feature, mask = self.encoder(img, img_mask)  # [b, t, d]
        if self.encoder_adapter is not None:
            feature = self.encoder_adapter(feature)
        return self.decoder.beam_search(
            [feature], [mask], beam_size, max_len, alpha, early_stopping, temperature
        )
