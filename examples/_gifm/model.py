from __future__ import annotations

# import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

# from neuralop.models import FNO

# from ppsci.arch.gino.neighbor_ops import NeighborMLPConvLayer
# from ppsci.arch.gino.neighbor_ops import NeighborMLPConvLayerLinear
# from ppsci.arch.gino.neighbor_ops import NeighborMLPConvLayerWeighted
# from ppsci.arch.gino.neighbor_ops import NeighborSearchLayer
# from ppsci.arch.gino.net_utils import MLP
# from ppsci.arch.gino.net_utils import AdaIN
# from ppsci.arch.gino.net_utils import PositionalEmbedding

try:
    from einops import rearrange
    from einops import repeat
except ModuleNotFoundError:
    pass
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple

from ppsci.arch import base
from ppsci.arch.cvit import CrossAttnBlock
from ppsci.arch.cvit import Mlp
from ppsci.arch.cvit import SelfAttnBlock
from ppsci.arch.mlp import FourierEmbedding
from ppsci.arch.mlp import PeriodEmbedding
from ppsci.utils import initializer
from ppsci.utils.misc import logger

# import jax
# import jax.numpy as paddle
# from jax.nn.initializers import uniform, normal, xavier_uniform

# import flax.linen as nn
# from typing import Optional, Callable, Dict, Union, Tuple


# Positional embedding from masked autoencoder https://arxiv.org/abs/2111.06377
def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: paddle.Tensor):
    if embed_dim % 2 != 0:
        raise ValueError(f"embedding dimension({embed_dim}) must be divisible by 2")

    omega = paddle.arange(embed_dim // 2, dtype=paddle.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape([-1])  # (M,)
    out = paddle.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = paddle.sin(out)  # (M, D/2)
    emb_cos = paddle.cos(out)  # (M, D/2)

    emb = paddle.concat([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim: int, length: int):
    return paddle.unsqueeze(
        get_1d_sincos_pos_embed_from_grid(
            embed_dim, paddle.arange(length, dtype=paddle.float32)
        ),
        0,
    )


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: Tuple[int, int]):
    def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
        if embed_dim % 2 != 0:
            raise ValueError(f"embedding dimension({embed_dim}) must be divisible by 2")

        # use half of dimensions to encode grid_h
        emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
        emb = paddle.concat([emb_h, emb_w], axis=1)  # (H*W, D)
        return emb

    grid_h = paddle.arange(grid_size[0], dtype=paddle.float32)
    grid_w = paddle.arange(grid_size[1], dtype=paddle.float32)
    grid = paddle.meshgrid(grid_w, grid_h, indexing="ij")  # here w goes first
    grid = paddle.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    return paddle.unsqueeze(pos_embed, 0)


class PatchEmbed(nn.Layer):
    def __init__(
        self,
        in_dim: int,
        spatial_dims: Sequence[int],
        patch_size: Tuple[int, ...] = (16, 16),
        emb_dim: int = 768,
        use_norm: bool = False,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.use_norm = use_norm
        self.layer_norm_eps = layer_norm_eps
        self.conv = nn.Conv2D(
            in_dim,
            self.emb_dim,
            (self.patch_size[0], self.patch_size[1]),
            (self.patch_size[0], self.patch_size[1]),
            data_format="NHWC",
        )
        self.norm = (
            nn.LayerNorm(self.emb_dim, self.layer_norm_eps)
            if self.use_norm
            else nn.Identity()
        )
        self._init_weights()

    def _init_weights(self) -> None:
        initializer.xavier_uniform_(self.conv.weight)
        initializer.constant_(self.conv.bias, 0)

    def forward(self, x):
        b, h, w, c = x.shape
        x = self.conv(x)  # [B, L, C] --> [B, L/ps, self.emb_dim]
        x = x.reshape(
            [
                b,
                (h // self.patch_size[0]) * (w // self.patch_size[1]),
                self.emb_dim,
            ]
        )
        if self.use_norm:
            x = self.norm(x)
        return x


class PerciverBlock(nn.Layer):
    def __init__(
        self,
        emb_dim: int,
        depth: int,
        num_heads: int = 8,
        num_latents: int = 64,
        mlp_ratio: int = 1,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.depth = depth
        self.num_heads = num_heads
        self.num_latents = num_latents
        self.mlp_ratio = mlp_ratio
        self.layer_norm_eps = layer_norm_eps
        self.latents = self.create_parameter(
            [self.num_latents, self.emb_dim],
            default_initializer=nn.initializer.Normal(std=1e-2),
        )
        self.cross_attn_blocks = nn.LayerList(
            [
                CrossAttnBlock(
                    self.num_heads, self.emb_dim, self.mlp_ratio, self.layer_norm_eps
                )
                for _ in range(self.depth)
            ]
        )
        self.norm = nn.LayerNorm(self.emb_dim, self.layer_norm_eps)

    def forward(self, x: paddle.Tensor):
        latents = repeat(self.latents, "l d -> b l d", b=x.shape[0])  # (B, L', D)
        for i in range(self.depth):
            latents = self.cross_attn_blocks[i](latents, x)

        latents = self.norm(latents)
        return latents


class Encoder(base.Arch):
    def __init__(
        self,
        in_dim: int,
        patch_size: int,
        grid_size: Tuple,
        emb_dim: int,
        num_latents: int,
        depth: int,
        num_heads: int,
        mlp_ratio: int,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.emb_dim = emb_dim
        self.num_latents = num_latents
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.layer_norm_eps = layer_norm_eps

        self.patch_embedding = PatchEmbed(
            in_dim, grid_size, self.patch_size, self.emb_dim
        )

        h, w = grid_size
        pos_emb = get_2d_sincos_pos_embed(
            self.emb_dim, (h // self.patch_size[0], w // self.patch_size[1])
        )
        self.pos_emb = self.create_parameter(
            pos_emb.shape, default_initializer=nn.initializer.Assign(pos_emb)
        )
        self.perceive_block = PerciverBlock(
            emb_dim=self.emb_dim,
            depth=2,
            num_heads=self.num_heads,
            num_latents=self.num_latents,
        )
        self.norm1 = nn.LayerNorm(self.emb_dim, epsilon=self.layer_norm_eps)
        self.self_attn_blocks = nn.LayerList(
            [
                SelfAttnBlock(
                    self.num_heads,
                    self.emb_dim,
                    self.mlp_ratio,
                    self.layer_norm_eps,
                )
                for _ in range(self.depth)
            ]
        )
        self.norm2 = nn.LayerNorm(self.emb_dim, epsilon=self.layer_norm_eps)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        b, h, w, c = x.shape

        # Patch embedding
        x = self.patch_embedding(x)
        logger.debug(f"x.shape = {x.shape}")

        # Interpolate positional embeddings to match the input shape
        pos_emb_interp = self.pos_emb.reshape(
            [
                1,
                self.grid_size[0] // self.patch_size[0],
                self.grid_size[1] // self.patch_size[1],
                self.emb_dim,
            ]
        )
        logger.debug(f"pos_emb_interp.shape = {pos_emb_interp.shape}")
        pos_emb_interp = F.interpolate(
            pos_emb_interp,
            [h // self.patch_size[0], w // self.patch_size[1]],
            mode="bilinear",
            data_format="NHWC",
        )
        pos_emb_interp = rearrange(pos_emb_interp, "b h w d -> b (h w) d")
        x = x + pos_emb_interp

        # Embed into tokens of the same length as the latents
        x = self.perceive_block(x)
        x = self.norm1(x)

        # Transformer
        for _, block in enumerate(self.self_attn_blocks):
            x = block(x)
        x = self.norm2(x)
        return x


class Decoder(base.Arch):
    def __init__(
        self,
        in_dim: int,
        fourier_freq: float = 1.0,
        period: Optional[Dict[str, Tuple[float, bool]]] = None,
        dec_depth: int = 2,
        dec_num_heads: int = 8,
        dec_emb_dim: int = 256,
        mlp_ratio: int = 1,
        out_dim: int = 1,
        num_mlp_layers: int = 1,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.fourier_freq = fourier_freq
        self.period = period
        self.dec_depth = dec_depth
        self.dec_num_heads = dec_num_heads
        self.dec_emb_dim = dec_emb_dim
        self.mlp_ratio = mlp_ratio
        self.out_dim = out_dim
        self.num_mlp_layers = num_mlp_layers
        self.layer_norm_eps = layer_norm_eps

        if self.period is not None:
            self.period_embed = PeriodEmbedding(period)

        self.fourier_embed = FourierEmbedding(
            in_dim if self.period is None else in_dim * 2,
            self.dec_emb_dim,
            self.fourier_freq,
        )

        self.fc = nn.Linear(self.dec_emb_dim, self.dec_emb_dim)

        self.cross_attn_blocks = nn.LayerList(
            [
                CrossAttnBlock(
                    self.dec_num_heads,
                    self.dec_emb_dim,
                    self.mlp_ratio,
                    self.layer_norm_eps,
                    self.dec_emb_dim,
                    self.dec_emb_dim,
                )
                for _ in range(self.dec_depth)
            ]
        )
        self.block_norm = nn.LayerNorm(self.dec_emb_dim, self.layer_norm_eps)
        self.final_mlp = Mlp(
            self.num_mlp_layers,
            self.dec_emb_dim,
            self.out_dim,
            layer_norm_eps=self.layer_norm_eps,
        )

    def forward(self, x, coords):
        b, n, c = x.shape

        # Embed periodic boundary conditions if specified
        if self.period is True:
            # Hardcode the periodicity, assuming the domain is [0, 1]x[0, 1]
            coords = self.period_embed(coords)
        logger.debug(f"coords.shape = {coords.shape}")

        coords = self.fourier_embed(coords)
        coords = paddle.expand(coords, [b, *coords.shape[1:]])
        logger.debug(f"coords.shape = {coords.shape}")
        # coords = repeat(coords, "d -> b n d", n=1, b=b)

        x = self.fc(x)

        logger.debug(f"x.shape = {x.shape}")
        for i in range(self.dec_depth):
            coords = self.cross_attn_blocks[i](coords, x)

        x = self.block_norm(coords)
        x = self.final_mlp(x)

        return x
