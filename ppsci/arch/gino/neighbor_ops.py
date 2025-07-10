from __future__ import annotations

import unittest
from typing import Optional

# import numpy as np
import open3d.ml.paddle as ml3d
import paddle
from paddle import nn
from paddle_scatter import segment_csr

from ppsci.arch.gino.net_utils import MLP

ENABLE_FUSED_SEGMENT_CSR = False
try:
    import fused_segment_csr

    ENABLE_FUSED_SEGMENT_CSR = True
except ModuleNotFoundError:
    pass


# def segment_mean_csr(src: paddle.Tensor, indptr: paddle.Tensor, out):
#     num_seg = indptr.shape[0] - 1
#     segment_ids = paddle.arange(num_seg)
#     repeats = indptr[1:] - indptr[:-1]
#     segment_ids = paddle.repeat_interleave(segment_ids, repeats,)
#     res = paddle.geometric.segment_mean(src, segment_ids)

#     if res.shape[0] < num_seg:
#         zero = paddle.zeros([num_seg - res.shape[0], res.shape[1]])
#         res = paddle.concat([res, zero], axis=0)
#     res = paddle.reshape(res, [num_seg, res.shape[1]])
#     return res


# def segment_sum_csr(src: paddle.Tensor, indptr: paddle.Tensor, out):
#     num_seg = indptr.shape[0] - 1
#     segment_ids = paddle.arange(num_seg)
#     repeats = indptr[1:] - indptr[:-1]
#     segment_ids = paddle.repeat_interleave(segment_ids, repeats)
#     res = paddle.geometric.segment_sum(src, segment_ids)

#     if res.shape[0] < num_seg:
#         zero = paddle.zeros([num_seg - res.shape[0], res.shape[1]])
#         res = paddle.concat([res, zero], axis=0)
#     res = paddle.reshape(res, [num_seg, -1])
#     return res


# def segment_csr(
#     src: paddle.Tensor,
#     indptr: paddle.Tensor,
#     out: Optional[paddle.Tensor] = None,
#     reduce: str = "sum",
# ) -> paddle.Tensor:
#     if reduce == "mean":
#         return segment_mean_csr(src, indptr, out)
#     elif reduce == "sum":
#         return segment_sum_csr(src, indptr, out)
#     else:
#         raise NotImplementedError


# from paddle_typing import paddle.Tensor


NeighborSearchReturnType = ml3d.python.return_types.open3d_fixed_radius_search


# def segment_mean_csr(src: paddle.Tensor, indptr: paddle.Tensor, out):
#     num_seg = indptr.shape[0] - 1
#     belongs_to = paddle.arange(num_seg)
#     repeats = indptr[1:] - indptr[:-1]
#     belongs_to = paddle.repeat_interleave(
#         belongs_to,
#         repeats,
#     )
#     print(f"src.shape = {src.shape}, belongs_to.shape = {belongs_to.shape}")
#     res = paddle.geometric.segment_mean(src, belongs_to)

#     if res.shape[0] < num_seg:
#         zero = paddle.zeros([num_seg - res.shape[0], res.shape[1]])
#         res = paddle.concat([res, zero], axis=0)
#     res = paddle.reshape(res, [num_seg, res.shape[1]])
#     return res


# def segment_sum_csr(src: paddle.Tensor, indptr: paddle.Tensor, out):
#     num_seg = indptr.shape[0] - 1
#     belongs_to = paddle.arange(num_seg)
#     repeats = indptr[1:] - indptr[:-1]
#     belongs_to = paddle.repeat_interleave(
#         belongs_to,
#         repeats,
#     )
#     res = paddle.geometric.segment_sum(src, belongs_to)

#     if res.shape[0] < num_seg:
#         zero = paddle.zeros([num_seg - res.shape[0], res.shape[1]])
#         res = paddle.concat([res, zero], axis=0)
#     res = paddle.reshape(res, [num_seg, -1])
#     return res


# def segment_csr(
#     src: paddle.Tensor,
#     indptr: paddle.Tensor,
#     out: Optional[paddle.Tensor] = None,
#     reduce: str = "sum",
# ) -> paddle.Tensor:
#     if reduce == "mean":
#         return segment_mean_csr(src, indptr, out)
#     elif reduce == "sum":
#         return segment_sum_csr(src, indptr, out)
#     else:
#         raise NotImplementedError


class NeighborSearchLayer(nn.Layer):
    def __init__(self, radius: float):
        super().__init__()
        self.radius = radius
        self.nsearch = ml3d.layers.FixedRadiusSearch()

    def forward(
        self, inp_positions: paddle.Tensor["N", 3], out_positions: paddle.Tensor["M", 3]
    ) -> NeighborSearchReturnType:
        # Convert precision to 'full' in advance
        # because 'half' is not supportted by FixedRadiusSearch().
        inp_positions = inp_positions.to(paddle.float32)
        out_positions = out_positions.to(paddle.float32)
        neighbors = self.nsearch(inp_positions, out_positions, self.radius)
        return neighbors


# class NeighborPoolingLayer(nn.Layer):
#     def __init__(self, reduction="mean"):
#         super().__init__()
#         self.reduction = reduction

#     def forward(
#         self, in_features: paddle.Tensor["N", "C"], neighbors: NeighborSearchReturnType
#     ) -> paddle.Tensor["M", "C"]:
#         """
#         inp_positions: [N,3]
#         out_positions: [M,3]
#         inp_features: [N,C]
#         neighbors: ml3d.layers.FixedRadiusSearchResult. If None, will be computed.
#         For the same inp_positions and out_positions, this can be reused.
#         """
#         rep_features = in_features[neighbors.neighbors_index.to("int64")]
#         out_features = segment_csr(
#             rep_features, neighbors.neighbors_row_splits, reduce=self.reduction
#         )
#         return out_features


class NeighborMLPConvLayer(nn.Layer):
    def __init__(
        self, mlp=None, in_channels=8, hidden_dim=32, out_channels=32, reduction="mean"
    ):
        super().__init__()
        self.reduction = reduction
        if mlp is None:
            mlp = MLP([2 * in_channels, hidden_dim, out_channels], nn.GELU)
        self.mlp = mlp

    def forward(
        self,
        in_features: paddle.Tensor["N", "C_in"],
        neighbors: NeighborSearchReturnType,
        out_features: Optional[paddle.Tensor["M", "C_in"]] = None,
    ) -> paddle.Tensor["M", "C_out"]:
        """
        inp_features: [N,C]
        outp_features: [M,C]
        neighbors: ml3d.layers.FixedRadiusSearchResult.
        """
        if out_features is None:
            out_features = in_features

        assert (
            in_features.shape[1] + out_features.shape[1]
            == self.mlp.layers[0].weight.shape[0]
        )
        rep_features = in_features[neighbors.neighbors_index.to("int64")]
        rs = neighbors.neighbors_row_splits
        num_reps = rs[1:] - rs[:-1]
        # repeat the self features using num_reps
        self_features = paddle.repeat_interleave(out_features, num_reps, axis=0)
        agg_features = paddle.concat([rep_features, self_features], axis=1)
        rep_features = self.mlp(agg_features)
        out_features = segment_csr(
            rep_features, neighbors.neighbors_row_splits, reduce=self.reduction
        )
        return out_features


class NeighborMLPConvLayerWeighted(nn.Layer):
    def __init__(
        self, mlp=None, in_channels=8, hidden_dim=32, out_channels=32, reduction="mean"
    ):
        super().__init__()
        self.reduction = reduction
        if mlp is None:
            mlp = MLP([2 * in_channels, hidden_dim, out_channels], nn.GELU)
        self.mlp = mlp

    ########################################################
    # @paddle.no_grad()
    ########################################################
    def forward(
        self,
        in_features: paddle.Tensor["N", "C_in"],
        neighbors: NeighborSearchReturnType,
        out_features: Optional[paddle.Tensor["M", "C_in"]] = None,
        in_weights: Optional[paddle.Tensor["N"]] = None,
    ) -> paddle.Tensor["M", "C_out"]:
        """
        in_features: [N,C]
        out_features: [M,C]
        in_weights: [N]
        neighbors: ml3d.layers.FixedRadiusSearchResult.
        """
        if out_features is None:
            out_features = in_features

        assert (
            in_features.shape[1] + out_features.shape[1]
            == self.mlp.layers[0].weight.shape[0]
        ), f"in_features.shape[1]({in_features.shape[1]}) + out_features.shape[1]({out_features.shape[1]}) != self.mlp.layers[0].weight.shape[0]({self.mlp.layers[0].weight.shape[0]})"

        # 每个点周围的邻居特征平铺: [*[grid0的邻居特征], *[grid0的邻居特征], ..., *[gridM的邻居特征]]
        if in_weights is None:
            rep_weights = 1
        else:
            rep_weights = in_weights[
                neighbors.neighbors_index.to(paddle.int64)
            ].unsqueeze(-1)

        rs = neighbors.neighbors_row_splits
        num_reps = rs[1:] - rs[:-1]
        # if ENABLE_FUSED_SEGMENT_CSR:
        #     neighbors_index_int64 = neighbors.neighbors_index.astype(paddle.int64)
        #     rep_csr = fused_segment_csr.select_segment_csr(
        #         in_features,
        #         neighbors_index_int64,
        #         neighbors.neighbors_row_splits,
        #         reduce=self.reduction,
        #     )
        # else:
        #     rep_features = in_features[neighbors.neighbors_index.to(paddle.int64)]
        #     # repeat the self features using num_reps

        #     # sum first and then mlp to make cuda memory usage only related to grid shape
        #     # 每个grid点各自将自己邻居特征聚合(求平均），得到m个平均特征
        #     rep_csr = segment_csr(
        #         rep_features, neighbors.neighbors_row_splits, reduce=self.reduction
        #     )
        #     del rep_features  # remove intermediate variables

        # if not self.training:
        #     self_csr = out_features.masked_fill(num_reps.unsqueeze(-1) == 0.0, 0.0)
        # else:
        #     self_features = paddle.repeat_interleave(out_features, num_reps, axis=0)
        #     self_csr = segment_csr(
        #         self_features, neighbors.neighbors_row_splits, reduce=self.reduction
        #     )
        #     del self_features
        # # assert len(out_features) == len(num_reps), f"out_features.shape = {out_features.shape}, num_reps.shape = {num_reps.shape}"
        # # print(out_features.shape, num_reps.shape)
        # # self_csr = out_features.masked_fill(num_reps.unsqueeze(-1) == 0, 0.0)
        # # self_csr = out_features
        # # 把grid邻居的平均特征和grid自己的特征concat
        # agg_csr = paddle.concat([rep_csr, self_csr], axis=1)
        # del rep_csr
        # del self_csr
        # # 每个grid点各自将自己邻居权重聚合(求平均），得到m个平均权重
        # # if in_weights is not None:
        # weights_csr = segment_csr(
        #     rep_weights, neighbors.neighbors_row_splits, reduce=self.reduction
        # )
        # del rep_weights
        # # else:
        # # weights_csr = 1
        # # 计算出每个grid的输出特征，即权重*mlp( concat(grid自己特征，平均特征) )
        # out_features = weights_csr * self.mlp(agg_csr)

        #### 原实现 ####
        # repeat the self features using num_reps
        self_features = paddle.repeat_interleave(out_features, num_reps, axis=0)
        agg_features = paddle.concat([rep_features, self_features], axis=1)
        rep_features = rep_weights * self.mlp(agg_features)
        out_features = segment_csr(
            rep_features, neighbors.neighbors_row_splits, reduce=self.reduction
        )
        ###############
        return out_features


class NeighborMLPConvLayerLinear(nn.Layer):
    def __init__(
        self, mlp=None, in_channels=8, hidden_dim=32, out_channels=32, reduction="mean"
    ):
        super().__init__()
        # if linear_kernel=False
        #   out_features = sum k([in_features, out_features])
        # if linear_kernel=True
        #   out_features = sum k([x_in, x_out]) * in_features

        self.reduction = reduction
        if mlp is None:
            mlp = MLP([2 * in_channels, hidden_dim, out_channels], nn.GELU)
        self.mlp = mlp

    ########################################################
    # @paddle.no_grad()
    ########################################################
    def forward(
        self,
        x_in: paddle.Tensor["N", "3"],
        neighbors: NeighborSearchReturnType,
        in_features: paddle.Tensor["N", "C"],
        x_out: Optional[paddle.Tensor["M", "3"]] = None,
    ) -> paddle.Tensor["M", "C_out"]:
        """
        inp_features: [N,C]
        outp_features: [M,C]
        neighbors: ml3d.layers.FixedRadiusSearchResult.
        """
        if x_out is None:
            x_out = x_in

        assert x_in.shape[1] + x_out.shape[1] == self.mlp.layers[0].weight.shape[0]
        # 每个点(点云点)周围的邻居(grid点)位置编码平铺: [*[点云0的邻居们的位置编码], *[点云1的邻居们的位置编码], ..., *[点云N的邻居们的位置编码]]
        rep_features = x_in[neighbors.neighbors_index]
        # 每个点(点云点)周围的邻居(grid点)特征平铺:     [*[点云0的邻居们的特征], *[点云1的邻居们的特征], ..., *[点云N的邻居们的特征]]
        in_features = in_features[neighbors.neighbors_index]
        rs = neighbors.neighbors_row_splits
        num_reps = rs[1:] - rs[:-1]
        # repeat the self features using num_reps
        # 每个点云特征各自复制邻居次数，与每个grid邻居特征一一对应
        self_features = paddle.repeat_interleave(x_out, num_reps, axis=0)
        # 将每个点云周围的grid特征和点云自己的特征concat
        agg_features = paddle.concat([rep_features, self_features], axis=1)
        # 过一遍mlp(算出权重？)
        del self_features

        """
        注: 此处尝试改为先聚合再mlp, 节省显存
        """
        ###### 原实现 #######
        rep_features = self.mlp(agg_features)  # (N, C)
        # 以rep_features为权重对每个点云各自的邻居grid特征加权
        rep_features = rep_features * in_features  # (N, C) * (N, C) -> (N, C)

        # 每个点云各自聚合加权完的特征
        out_features = segment_csr(
            rep_features, neighbors.neighbors_row_splits, reduce=self.reduction
        )
        ############################

        ###### 节省显存的实现 #######
        # agg_features = segment_csr(
        #     agg_features, neighbors.neighbors_row_splits, reduce=self.reduction
        # )
        # in_features = segment_csr(
        #     in_features, neighbors.neighbors_row_splits, reduce=self.reduction
        # )
        # rep_features = self.mlp(agg_features)  # (N, C)
        # del agg_features
        # # 以rep_features为权重对每个点云各自的邻居grid特征加权
        # out_features = rep_features * in_features  # (N, C) * (N, C) -> (N, C)
        ############################

        return out_features


class TestNeighborSearch(unittest.TestCase):
    def setUp(self) -> None:
        self.N = 10000
        # self.device = "cuda:0"
        return super().setUp()

    def test_neighbor_search(self):
        pass
        # inp_positions = paddle.randn([self.N, 3]) * 10
        # inp_features = paddle.randn([self.N, 8])
        # out_positions = inp_positions

        # neighbors = NeighborSearchLayer(1.2)(inp_positions, out_positions)
        # pool = NeighborPoolingLayer(reduction="mean")
        # out_features = pool(inp_features, neighbors)

    def test_mlp_conv(self):
        out_N = 1000
        radius = 1.2
        in_positions = paddle.randn([self.N, 3]) * 10
        out_positions = paddle.randn([out_N, 3]) * 10
        in_features = paddle.randn([self.N, 8])
        out_features = paddle.randn([out_N, 8])

        neighbors = NeighborSearchLayer(radius)(in_positions, out_positions)
        conv = NeighborMLPConvLayer(reduction="mean")
        out_features = conv(in_features, neighbors, out_features=out_features)


if __name__ == "__main__":
    unittest.main()
