from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from typing import Optional

import paddle
from omegaconf import DictConfig
from paddle.nn import functional as F


def shallow_asdict(obj) -> dict:
    if isinstance(obj, DictConfig):
        return {k: v for k, v in obj.items()}
    else:
        raise TypeError(f"Unsupported type for shallow_asdict: {type(obj)}")


class ConditionedNorm(paddle.nn.Layer):
    """
    Used for time-conditioned Layer Normalization.

    Parameters
    ----------
    input_size:int
        The size of the input tensor
    output_size:int
        The size of the output tensor
    hidden_size:int
        The size of the hidden layer
    """

    def __init__(self, input_size: int, output_size: int, hidden_size: int):
        super().__init__()
        self.mlp_scale = MLP(
            input_size, output_size, hidden_size, num_layers=2, activation="none"
        )
        self.mlp_bias = MLP(
            input_size, output_size, hidden_size, num_layers=2, activation="none"
        )
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.mlp_scale.layers:
            paddle.nn.init.normal_(layer.weight, std=0.01)
        for layer in self.mlp_bias.layers:
            paddle.nn.init.normal_(layer.weight, std=0.01)

    def forward(self, c, x):
        """
        For time-conditioned Layer Normalization, the c is often scalar time difference $	au$,
        x is a tensor. The output of the mlp_scale and mlp_bias are a vector which have the same size as x.

        Parameters
        ----------
        c:paddle.Tensor
            The conditional input, often time.
            [batch_size, 1]
        x:paddle.Tensor
            The input tensor
        """
        scale = 1 + c * self.mlp_scale(c)
        bias = c * self.mlp_bias(c)
        x = x * scale[:, None, :] + bias[:, None, :]
        return x


@dataclass
class AugmentedMLPConfig:
    hidden_size: int = 64
    num_layers: int = 3
    activation: str = "swish"
    use_layer_norm: bool = True
    use_conditional_norm: bool = False
    cond_norm_hidden_size: int = 4


def activation_fn(name: str, activation_kwargs: dict = dict()) -> Callable:
    if name == "none":
        return lambda x: x
    elif name == "swish":
        return paddle.nn.SiLU(**activation_kwargs)
    elif hasattr(paddle.nn.functional, name):
        return getattr(F, name)
    else:
        raise ValueError(f"Activation function {name} not found")


class MLP(paddle.nn.Layer):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        num_layers: int = 3,
        activation: str = "swish",
    ):
        super().__init__()
        if num_layers <= 2:
            self.layers = paddle.nn.LayerList(
                sublayers=[
                    paddle.nn.Linear(in_features=input_size, out_features=output_size)
                ]
            )
        else:
            self.layers = paddle.nn.LayerList(
                sublayers=[
                    paddle.nn.Linear(in_features=input_size, out_features=hidden_size)
                ]
            )
            for _ in range(num_layers - 2):
                self.layers.append(
                    paddle.nn.Linear(in_features=hidden_size, out_features=hidden_size)
                )
            self.layers.append(
                paddle.nn.Linear(in_features=hidden_size, out_features=output_size)
            )
        self.act = activation_fn(activation)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.act(x)
        x = self.layers[-1](x)
        return x


class AugmentedMLP(paddle.nn.Layer):
    mlp: MLP
    norm: Optional[paddle.nn.LayerNorm]
    correction: Optional[ConditionedNorm]
    input_size: int
    output_size: int
    hidden_size: int
    """
    This is a simple MLP with layer normalization and conditional normalization.

    Parameters
    ----------
    input_size:int
        The size of the input tensor
    output_size:int
        The size of the output tensor
    hidden_size:int
        The size of the hidden layer
    num_layers:int
        The number of layers
    activation:str
        The activation function
    use_layer_norm:bool
        Whether to use layer normalization. Default to True
    use_conditional_norm:bool
        Whether to use conditional normalization. Default to False. Note that if True,
        the conditional input should be provided and the correction should be provided.
        use_layer_norm should be False.
    cond_norm_hidden_size:int
        The size of the hidden layer for conditional normalization. Default to 4
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 64,
        num_layers: int = 3,
        activation: str = "swish",
        use_layer_norm: bool = True,
        use_conditional_norm: bool = False,
        cond_norm_hidden_size: int = 4,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.mlp = MLP(input_size, output_size, hidden_size, num_layers, activation)
        if use_layer_norm:
            self.norm = paddle.nn.LayerNorm(normalized_shape=output_size)
        else:
            self.norm = None
        if use_conditional_norm:
            self.correction = ConditionedNorm(1, output_size, cond_norm_hidden_size)
        else:
            self.correction = None
        self.reset_parameters()

    def reset_parameters(self):
        self.mlp.reset_parameters()

    def forward(
        self, x: paddle.Tensor, condition: Optional[float] = None
    ) -> paddle.Tensor:
        """
        Parameters
        ----------
        x: paddle.Tensor
            ND Tensor of shape [..., input_size]
        condition: Optional[float]

        Returns
        -------
        x: paddle.Tensor
            ND Tensor of shape [..., output_size]
        """
        x = self.mlp(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.correction is not None:
            assert condition is not None, "Conditional input c should be provided"
            x = self.correction(c=condition, x=x)
        return x

    @classmethod
    def from_config(cls, input_size: int, output_size: int, config: AugmentedMLPConfig):
        return cls(input_size, output_size, **shallow_asdict(config))


class ChannelMLP(paddle.nn.Layer):
    """ChannelMLP applies an arbitrary number of layers of
    1d convolution and nonlinearity to the channels of input
    and is invariant to spatial resolution.

    Parameters
    ----------
    in_channels : int
    out_channels : int, default is None
        if None, same is in_channels
    hidden_channels : int, default is None
        if None, same is in_channels
    n_layers : int, default is 2
        number of linear layers in the MLP
    non_linearity : default is F.gelu
    dropout : float, default is 0
        if > 0, dropout probability
    """

    def __init__(
        self,
        in_channels,
        out_channels=None,
        hidden_channels=None,
        n_layers=2,
        n_dim=2,
        non_linearity=paddle.nn.functional.gelu,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.hidden_channels = (
            in_channels if hidden_channels is None else hidden_channels
        )
        self.non_linearity = non_linearity
        self.dropout = (
            paddle.nn.LayerList(
                sublayers=[paddle.nn.Dropout(p=dropout) for _ in range(n_layers)]
            )
            if dropout > 0.0
            else None
        )
        self.fcs = paddle.nn.LayerList()
        for i in range(n_layers):
            if i == 0 and i == n_layers - 1:
                self.fcs.append(
                    paddle.nn.Conv1d(self.in_channels, self.out_channels, 1)
                )
            elif i == 0:
                self.fcs.append(
                    paddle.nn.Conv1d(self.in_channels, self.hidden_channels, 1)
                )
            elif i == n_layers - 1:
                self.fcs.append(
                    paddle.nn.Conv1d(self.hidden_channels, self.out_channels, 1)
                )
            else:
                self.fcs.append(
                    paddle.nn.Conv1d(self.hidden_channels, self.hidden_channels, 1)
                )

    def forward(self, x):
        reshaped = False
        size = list(x.shape)
        if x.ndim > 3:
            x = x.reshape((*size[:2], -1))
            reshaped = True
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < self.n_layers - 1:
                x = self.non_linearity(x)
            if self.dropout is not None:
                x = self.dropout[i](x)
        if reshaped:
            x = x.reshape((size[0], self.out_channels, *size[2:]))
        return x


class LinearChannelMLP(paddle.nn.Layer):
    def __init__(self, layers, non_linearity=paddle.nn.functional.gelu, dropout=0.0):
        super().__init__()
        self.n_layers = len(layers) - 1
        assert self.n_layers >= 1
        self.fcs = paddle.nn.LayerList()
        self.non_linearity = non_linearity
        self.dropout = (
            paddle.nn.LayerList(
                sublayers=[paddle.nn.Dropout(p=dropout) for _ in range(self.n_layers)]
            )
            if dropout > 0.0
            else None
        )
        for j in range(self.n_layers):
            self.fcs.append(
                paddle.nn.Linear(in_features=layers[j], out_features=layers[j + 1])
            )

    def forward(self, x):
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < self.n_layers - 1:
                x = self.non_linearity(x)
            if self.dropout is not None:
                x = self.dropout[i](x)
        return x
