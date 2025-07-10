import paddle
import paddle.nn as nn


# A simple feedforward neural network
class MLP(nn.Layer):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super().__init__()
        self.n_layers = len(layers) - 1
        assert self.n_layers >= 1

        self.layers = nn.LayerList()
        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j + 1]))
            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm2D(layers[j + 1]))
                self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)
        return x


class PositionalEmbedding(nn.Layer):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = paddle.arange(start=0, end=self.num_channels // 2, dtype=paddle.float32)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.outer(freqs.to(x.dtype))
        x = paddle.concat([x.cos(), x.sin()], axis=1)
        return x


class AdaIN(nn.Layer):
    def __init__(self, embed_dim, in_channels, mlp=None, eps=1e-5, len_info: int = 5):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.eps = eps

        if mlp is None:
            mlp = nn.Sequential(
                nn.Linear(embed_dim * len_info, 512),
                nn.GELU(),
                nn.Linear(512, 2 * in_channels),
            )
        self.mlp = mlp

        self.embedding = None

    def update_embeddding(self, x):
        self.embedding = x.reshape([-1])

    def forward(self, x):
        assert (
            self.embedding is not None
        ), "AdaIN: update embeddding before running forward"
        tmp = self.mlp(self.embedding)
        weight, bias = paddle.split(tmp, tmp.shape[0] // self.in_channels, axis=0)

        return nn.functional.group_norm(x, self.in_channels, self.eps, weight, bias)
