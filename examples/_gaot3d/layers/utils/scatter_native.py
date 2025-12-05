from __future__ import annotations

from typing import Optional

import paddle


def scatter_native(
    src: paddle.Tensor,
    index: paddle.Tensor,
    dim: int = -1,
    out: Optional[paddle.Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> paddle.Tensor:
    if dim != 0:
        raise NotImplementedError("Native scatter fallback only supports dim=0")
    if dim_size is None:
        dim_size = int(index._max()) + 1 if index.size > 0 else 0
    shape = list(src.shape)
    shape[dim] = dim_size
    if out is None:
        out = paddle.zeros(shape, dtype=src.dtype, device=src.place)
    else:
        out = out.fill_(0.0)
    index_expanded = index.view([-1] + [1] * (src.dim() - 1)).expand_as(src)
    if reduce == "sum" or reduce == "add":
        return out.scatter_add_(dim, index_expanded, src)
    elif reduce == "mean":
        out_sum = out.scatter_add_(dim, index_expanded, src)
        counts = (
            paddle.bincount(x=index, minlength=dim_size).to(src.dtype).to(src.place)
        )
        count_shape = [1] * src.dim()
        count_shape[dim] = dim_size
        counts = counts.view(count_shape)
        return out_sum / counts.clamp(min=1)
    elif reduce == "max" or reduce == "amax":
        if hasattr(out, "scatter_reduce_"):
            """Not Support auto convert *.scatter_reduce_, please judge whether it is Pytorch API and convert by yourself"""
            out.scatter_reduce_(
                dim, index_expanded, src, reduce="amax", include_self=False
            )
            return out
        else:
            print(
                "Warning: Native 'max' reduction is approximate without scatter_reduce_."
            )
            return out.scatter_add_(dim, index_expanded, src)
    elif reduce == "min" or reduce == "amin":
        if hasattr(out, "scatter_reduce_"):
            out.fill_(float("inf"))
            """Not Support auto convert *.scatter_reduce_, please judge whether it is Pytorch API and convert by yourself"""
            out.scatter_reduce_(
                dim, index_expanded, src, reduce="amin", include_self=False
            )
            out = paddle.where(out == float("inf"), 0.0, out)
            return out
        else:
            print(
                "Warning: Native 'min' reduction not fully implemented without scatter_reduce_."
            )
            return out.scatter_add_(dim, index_expanded, src)
    else:
        raise ValueError(f"Unsupported reduce operation '{reduce}' in native scatter")
