import torch
from torch import nn
from torch.nn import functional as F


class InLayer(nn.Module):
    def __init__(
        self,
        categorical_cols: list[str],
        categorical_sizes: dict[str, int],
        numerical_cols: list[str] = [],
        embedding_size: int = 768,
        strategy: str = "concat",
        padding_idx: int = 0,
    ):
        assert len(categorical_cols) == len(categorical_sizes)

        super(InLayer, self).__init__()

        self.embedding_size = embedding_size
        self.categorical_cols = categorical_cols
        self.categorical_sizes = categorical_sizes
        self.numerical_cols = numerical_cols
        self.padding_idx = padding_idx
        self.strategy = strategy

        self.total_features = len(categorical_cols) + len(numerical_cols)

        if strategy == "concat":
            in_embedding_size = embedding_size // 2
        elif strategy == "sum":
            in_embedding_size = embedding_size
        else:
            raise ValueError("Invalid strategy")

        # assert embedding size is divisible by the number of features
        # assert embedding_size % self.total_features == 0, "Embedding size must be divisible by the number of features"
        self.embedding_layers = nn.ModuleDict()
        for col in categorical_cols:
            self.embedding_layers[col] = nn.Embedding(
                categorical_sizes[col],
                in_embedding_size,
                padding_idx=padding_idx,
            )

        if len(numerical_cols) > 0:
            self.continuous_layer = nn.Linear(len(numerical_cols), in_embedding_size)

        self.layer_norm = nn.LayerNorm(embedding_size)
        self.init_params()

    def forward(self, cat_x, num_x=None):

        # cat features
        embedded_features = []
        for ix, name in enumerate(self.categorical_cols):
            # since we use OrderedDict, we can access the embedding layer by index
            embed = self.embedding_layers[name](cat_x[..., ix])
            embedded_features.append(embed)

        # num features
        if len(self.numerical_cols) > 0:
            projected_features = self.continuous_layer(num_x)

        # concatenate or sum
        if self.strategy == "concat":
            x = torch.cat(embedded_features, dim=-1)
            if len(self.numerical_cols) > 0:
                x = torch.cat(
                    [x, projected_features],
                    dim=-1,
                )
        elif self.strategy == "sum":
            x = sum(embedded_features)
            if len(self.numerical_cols) > 0:
                x += projected_features

        x = self.layer_norm(x)
        return x

    def init_params(self):
        for _, layer in self.embedding_layers.items():
            nn.init.xavier_uniform_(layer.weight)

        if len(self.numerical_cols) > 0:
            nn.init.xavier_uniform_(self.continuous_layer.weight)


class OutLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(OutLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.layer_norm = nn.LayerNorm(input_size)
        self.linear = nn.Linear(input_size, output_size)
        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x = self.layer_norm(x)
        return self.linear(x)
