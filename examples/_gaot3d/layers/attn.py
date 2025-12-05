from __future__ import annotations

from typing import Any
from typing import Optional

import paddle
from layers.mlp import ConditionedNorm
from layers.rotary_embedding_paddle import RotaryEmbedding
from omegaconf import DictConfig
from omegaconf import OmegaConf
from paddle import nn


def shallow_asdict(obj: Any) -> dict:
    if isinstance(obj, DictConfig):
        return {k: v for k, v in obj.items()}
    else:
        raise TypeError(f"Unsupported type for shallow_asdict: {type(obj)}")


# 定义子配置
AttentionConfig = OmegaConf.create(
    {
        "hidden_size": 256,
        "num_heads": 8,
        "num_kv_heads": 8,
        "use_conditional_norm": False,
        "cond_norm_hidden_size": 4,
        "atten_dropout": 0.1,
        "positional_embedding": "absolute",
        "H": None,
        "W": None,
    }
)

FFNConfig = OmegaConf.create(
    {
        "hidden_size": 1024,
        "use_conditional_norm": False,
        "cond_norm_hidden_size": 4,
    }
)

TransformerConfig = OmegaConf.create(
    {
        "patch_size": 2,
        "hidden_size": 256,
        "use_attn_norm": True,
        "use_ffn_norm": True,
        "norm_eps": 1e-6,
        "num_layers": 10,
        "positional_embedding": "rope",
        "use_long_range_skip": True,
        "attn_config": AttentionConfig,
        "ffn_config": FFNConfig,
    }
)

"""
Reference: https://github.com/meta-llama/llama3/blob/main/llama/model.py
"""


class GroupQueryFlashAttention(nn.Layer):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 128,
        num_heads: int = 8,
        num_kv_heads: int = 4,
        use_conditional_norm: bool = False,
        cond_norm_hidden_size: int = 4,
        atten_dropout: float = 0.0,
        H: int = 64,
        W: int = 64,
        positional_embedding: str = "absolute",
    ):
        super().__init__()
        assert (
            hidden_size % num_heads == 0
        ), f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
        assert (
            num_heads % num_kv_heads == 0
        ), f"num_heads {num_heads} must be divisible by num_kv_heads {num_kv_heads}"
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_repeat = num_heads // num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.atten_dropout = atten_dropout
        kv_hidden_size = self.head_dim * self.num_kv_heads
        self.q_proj = nn.Linear(
            in_features=input_size, out_features=hidden_size, bias_attr=False
        )
        self.k_proj = nn.Linear(
            in_features=input_size, out_features=kv_hidden_size, bias_attr=False
        )
        self.v_proj = nn.Linear(
            in_features=input_size, out_features=kv_hidden_size, bias_attr=False
        )
        self.o_proj = nn.Linear(
            in_features=hidden_size, out_features=output_size, bias_attr=False
        )
        if use_conditional_norm:
            self.correction = ConditionedNorm(1, output_size, cond_norm_hidden_size)
        else:
            self.correction = None
        self.attn_dtype = paddle.float16
        if positional_embedding == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.head_dim)

    def forward(
        self,
        x: paddle.Tensor,
        condition: Optional[float] = None,
        relative_positions: Optional[paddle.Tensor] = None,
    ):
        """
        Parameters
        ----------
        x: torch.Tensor, shape (..., seq_len, input_size)

        Returns
        -------
        torch.Tensor, shape (..., seq_len, output_size)
        """
        if self.correction is not None:
            x = self.correction(c=condition, x=x)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        batch_size, seq_len, _ = q.size()
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(
            1, 2
        )
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(
            1, 2
        )
        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_repeat, dim=1)
            v = v.repeat_interleave(self.num_repeat, dim=1)
        if relative_positions is not None:
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)
        if self.training:
            dp = self.atten_dropout
        else:
            dp = 0.0
        x = nn.functional.scaled_dot_product_attention(
            q.transpose([0, 2, 1, 3]),
            k.transpose([0, 2, 1, 3]),
            v.transpose([0, 2, 1, 3]),
            attn_mask=None,
            dropout_p=dp,
        ).transpose([0, 2, 1, 3])
        x = x.transpose(1, 2).reshape(batch_size, seq_len, -1)
        x = self.o_proj(x)
        return x

    @classmethod
    def from_config(cls, input_size: int, output_size: int, config: AttentionConfig):
        config = OmegaConf.to_container(config, resolve=True)
        config.pop("D", None)
        config = OmegaConf.create(config)
        return cls(input_size, output_size, **config)


class FFN(nn.Layer):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 256,
        use_conditional_norm: bool = False,
        cond_norm_hidden_size: int = 4,
    ):
        super().__init__()
        self.w1 = nn.Linear(
            in_features=input_size, out_features=hidden_size, bias_attr=False
        )
        self.w2 = nn.Linear(
            in_features=hidden_size, out_features=output_size, bias_attr=False
        )
        self.w3 = nn.Linear(
            in_features=input_size, out_features=hidden_size, bias_attr=False
        )
        if use_conditional_norm:
            self.correction = ConditionedNorm(1, output_size, cond_norm_hidden_size)
        else:
            self.correction = None

    def forward(self, x, condition: Optional[float] = None):
        x = self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))
        if self.correction is not None:
            x = self.correction(c=condition, x=x)
        return x

    @classmethod
    def from_config(cls, input_size: int, output_size: int, config: FFNConfig):
        return cls(input_size, output_size, **shallow_asdict(config))


class RMSNorm(nn.Layer):
    def __init__(self, dim: int, eps: float = 1e-06):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(paddle.ones(dim))

    def _norm(self, x):
        return x * paddle.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class TransformerBlock(nn.Layer):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        use_attn_norm: bool = True,
        use_ffn_norm: bool = True,
        norm_eps: float = 1e-06,
        attn_config: AttentionConfig = AttentionConfig,
        ffn_config: FFNConfig = FFNConfig,
        skip_connection: bool = False,
    ):
        super().__init__()
        self.attn = GroupQueryFlashAttention.from_config(
            input_size, attn_config.hidden_size, config=attn_config
        )
        self.ffn = FFN.from_config(
            attn_config.hidden_size, output_size, config=ffn_config
        )
        self.attn_norm = RMSNorm(input_size, eps=norm_eps) if use_attn_norm else None
        self.ffn_norm = (
            RMSNorm(attn_config.hidden_size, eps=norm_eps) if use_ffn_norm else None
        )
        self.skip_connection = skip_connection
        if self.skip_connection:
            self.skip_proj = nn.Linear(
                in_features=input_size + output_size, out_features=input_size
            )

    def forward(
        self,
        x: paddle.Tensor,
        condition: Optional[float] = None,
        relative_positions: Optional[paddle.Tensor] = None,
        skip: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        """
        Parameters
        ----------
        x: torch.Tensor, shape (..., seq_len, input_size)
        condition:Optional[float]

        Returns
        -------
        torch.Tensor, shape (..., seq_len, output_size)
        """
        if self.skip_connection and skip is not None:
            x = paddle.cat([x, skip], dim=-1)
            x = self.skip_proj(x)
        h = x if self.attn_norm is None else self.attn_norm(x)
        h = x + self.attn(h, condition=condition, relative_positions=relative_positions)
        h = h if self.ffn_norm is None else self.ffn_norm(h)
        out = h + self.ffn(h, condition=condition)
        return out

    @classmethod
    def from_config(
        cls,
        input_size: int,
        output_size: int,
        skip_connection: bool = False,
        config: TransformerConfig = TransformerConfig,
    ):
        config.attn_config.positional_embedding = config.positional_embedding
        kwargs = shallow_asdict(config)
        kwargs.pop("num_layers")
        kwargs.pop("hidden_size")
        kwargs.pop("positional_embedding")
        kwargs.pop("use_long_range_skip")
        kwargs.pop("patch_size")
        # kwargs.pop("D")
        return cls(input_size, output_size, skip_connection=skip_connection, **kwargs)


class Transformer(nn.Layer):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        config: TransformerConfig = TransformerConfig,
    ):
        super().__init__()
        hidden_size: int = config.hidden_size
        num_layers: int = config.num_layers
        self.use_long_range_skip = config.use_long_range_skip
        if input_size != hidden_size:
            self.input_proj = nn.Linear(
                in_features=input_size, out_features=hidden_size
            )
        else:
            self.input_proj = nn.Identity()
        if hidden_size != output_size:
            self.output_proj = nn.Linear(
                in_features=hidden_size, out_features=output_size
            )
        else:
            self.output_proj = nn.Identity()
        num_encoder_layers = num_layers // 2
        num_decoder_layers = num_layers // 2
        middle_layer_exists = num_layers % 2 == 1
        self.encoder_layers = nn.LayerList(
            sublayers=[
                TransformerBlock.from_config(
                    input_size=hidden_size,
                    output_size=hidden_size,
                    skip_connection=False,
                    config=config,
                )
                for _ in range(num_encoder_layers)
            ]
        )
        self.middle_layer = None
        if middle_layer_exists:
            self.middle_layer = TransformerBlock.from_config(
                input_size=hidden_size,
                output_size=hidden_size,
                skip_connection=False,
                config=config,
            )
        self.decoder_layers = nn.LayerList(
            sublayers=[
                TransformerBlock.from_config(
                    input_size=hidden_size,
                    output_size=hidden_size,
                    skip_connection=True,
                    config=config,
                )
                for _ in range(num_decoder_layers)
            ]
        )

    def forward(
        self,
        x: paddle.Tensor,
        condition: Optional[float] = None,
        relative_positions: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        """
        Parameters
        ----------
        x: torch.Tensor
            [..., seq_len, input_size]

        Returns
        -------
        torch.Tensor
            [..., seq_len, output_size]
        """
        x = self.input_proj(x)
        skips = []
        for layer in self.encoder_layers:
            x = layer(x, condition=condition, relative_positions=relative_positions)
            skips.append(x)
        if self.middle_layer is not None:
            x = self.middle_layer(
                x, condition=condition, relative_positions=relative_positions
            )
        for layer in self.decoder_layers:
            skip = skips.pop() if self.use_long_range_skip else None
            x = layer(
                x, condition=condition, relative_positions=relative_positions, skip=skip
            )
        x = self.output_proj(x)
        return x
