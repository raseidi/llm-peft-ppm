import torch.nn as nn

from transformers import AutoModel
from peft import get_peft_model, LoraConfig

from ppm.models.common import InLayer, OutLayer
from ppm.models.config import FreezeConfig

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import os

HF_TOKEN = os.getenv("HF_TOKEN")


class NextEventPredictor(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        categorical_cols: list[str],
        categorical_sizes: dict[str, int],
        numerical_cols: list[str],
        categorical_targets: list[str],
        numerical_targets: list[str],
        padding_idx: int,
        strategy: str,
        backbone_name: str,
        backbone_pretrained: bool,
        backbone_finetuning: LoraConfig | FreezeConfig | None,
        backbone_type: str,
        backbone_hidden_size: int,
        backbone_n_layers: int,
        device: str,
    ):
        super(NextEventPredictor, self).__init__()

        self.categorical_cols = categorical_cols
        self.categorical_sizes = categorical_sizes
        self.numerical_cols = numerical_cols
        self.categorical_targets = categorical_targets
        self.numerical_targets = numerical_targets

        self.embedding_size = embedding_size
        self.strategy = strategy

        self.backbone_name = backbone_name
        self.backbone_pretrained = backbone_pretrained
        self.backbone_finetuning = backbone_finetuning
        self.backbone_type = backbone_type
        self.backbone_hidden_size = backbone_hidden_size
        self.backbone_n_layers = backbone_n_layers

        self.padding_idx = padding_idx
        self.device = device

        # define input layer
        self.in_layer = InLayer(
            # output size
            embedding_size=embedding_size,
            # input sizes
            categorical_cols=categorical_cols,
            categorical_sizes=categorical_sizes,
            numerical_cols=numerical_cols,
            # other params
            padding_idx=padding_idx,
            strategy=strategy,
        )

        # define backbone
        if backbone_pretrained:
            self.backbone = AutoModel.from_pretrained(backbone_name, token=HF_TOKEN)
            if isinstance(backbone_finetuning, LoraConfig):
                self.backbone = get_peft_model(self.backbone, backbone_finetuning)
            elif isinstance(backbone_finetuning, FreezeConfig):
                # self._freeze_params(backbone_finetuning)
                backbone_finetuning.apply(self.backbone)
            else:
                raise NotImplementedError("Fine-tuning not implemented yet.")
        else:
            if backbone_name == "rnn":
                if backbone_type == "lstm":
                    self.backbone = nn.LSTM
                elif backbone_type == "gru":
                    self.backbone = nn.GRU
                elif backbone_type == "rnn":
                    self.backbone = nn.RNN
                else:
                    raise ValueError("Invalid RNN type.")
                self.backbone = self.backbone(
                    input_size=embedding_size,
                    hidden_size=backbone_hidden_size,
                    num_layers=backbone_n_layers,
                    batch_first=True,
                )
            if backbone_name.endswith("gpt2"):
                self.backbone = AutoModel.from_pretrained(
                    "openai-community/gpt2",
                )
                self.backbone.apply(self.backbone._init_weights)

        # define output layer(s)
        self.out_layers = nn.ModuleDict()
        for target in categorical_targets:
            self.out_layers[target] = OutLayer(
                input_size=backbone_hidden_size,
                output_size=categorical_sizes[target.replace("next_", "")],
            )
        for target in numerical_targets:
            self.out_layers[target] = OutLayer(
                input_size=backbone_hidden_size,
                output_size=1,
            )

    def forward(self, x_cat, x_num=None, attention_mask=None, h=None):
        x = self.in_layer(x_cat, x_num)

        if self.backbone_name == "rnn":
            lengths = attention_mask.sum(dim=-1).long().tolist()
            if not isinstance(lengths, list):
                lengths = [lengths]
            x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            x = self.backbone(x, h)
        else:
            x = self.backbone(inputs_embeds=x, attention_mask=attention_mask)

        if hasattr(x, "last_hidden_state"):
            x = x.last_hidden_state
        elif isinstance(x, tuple):
            x, h = x
            h = tuple([h_.detach() for h_ in h])
            x = pad_packed_sequence(x, batch_first=True)[0]
        else:
            raise ValueError("Invalid output from backbone.")

        out = {}
        for target in self.out_layers:
            out[target] = self.out_layers[target](x)
        return out, h
