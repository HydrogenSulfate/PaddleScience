from __future__ import annotations

# import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from neuralop.models import FNO

from ppsci.arch.gino.neighbor_ops import NeighborMLPConvLayer
from ppsci.arch.gino.neighbor_ops import NeighborMLPConvLayerLinear
from ppsci.arch.gino.neighbor_ops import NeighborMLPConvLayerWeighted
from ppsci.arch.gino.neighbor_ops import NeighborSearchLayer
from ppsci.arch.gino.net_utils import MLP
from ppsci.arch.gino.net_utils import AdaIN
from ppsci.arch.gino.net_utils import PositionalEmbedding


class BaseModel(nn.Layer):
    def __init__(self):
        super().__init__()
        # self.device_indicator_param = nn.Parameter(paddle.empty(0))
        self.device_indicator_param = paddle.empty([0])

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.device_indicator_param.device

    def data_dict_to_input(self, data_dict, **kwargs):
        """
        Convert data dictionary to appropriate input for the model.
        """
        raise NotImplementedError

    def loss_dict(self, data_dict, **kwargs):
        """
        Compute the loss dictionary for the model.
        """
        raise NotImplementedError

    @paddle.no_grad()
    def eval_dict(self, data_dict, **kwargs):
        """
        Compute the evaluation dictionary for the model.
        """
        raise NotImplementedError


class Projection(nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=None,
        n_dim=2,
        non_linearity=F.gelu,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = (
            in_channels if hidden_channels is None else hidden_channels
        )
        self.non_linearity = non_linearity
        Conv = getattr(nn, f"Conv{n_dim}D")
        self.fc1 = Conv(in_channels, hidden_channels, 1)
        self.fc2 = Conv(hidden_channels, out_channels, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.non_linearity(x)
        x = self.fc2(x)
        return x


class GNOFNOGNO(BaseModel):
    def __init__(
        self,
        radius_in=0.05,
        radius_out=0.05,
        embed_dim=64,
        hidden_channels=(86, 86),
        in_channels=1,
        out_channels=1,
        fno_modes=(32, 32, 32),
        fno_hidden_channels=86,
        fno_out_channels=86,
        fno_domain_padding=0.125,
        fno_norm="group_norm",
        fno_factorization="tucker",
        fno_rank=0.4,
        linear_kernel=True,
        weighted_kernel=True,
    ):
        super().__init__()
        self.weighted_kernel = weighted_kernel
        self.nb_search_in = NeighborSearchLayer(radius_in)
        self.nb_search_out = NeighborSearchLayer(radius_out)
        self.pos_embed = PositionalEmbedding(embed_dim)
        self.df_embed = MLP([in_channels, embed_dim, 3 * embed_dim], paddle.nn.GELU)
        self.linear_kernel = linear_kernel
        kernel1 = MLP([10 * embed_dim, 512, 256, hidden_channels[0]], paddle.nn.GELU)
        self.gno1 = NeighborMLPConvLayerWeighted(mlp=kernel1)
        if not linear_kernel:
            kernel2 = MLP(
                [fno_out_channels + 4 * embed_dim, 512, 256, hidden_channels[1]],
                paddle.nn.GELU,
            )
            self.gno2 = NeighborMLPConvLayer(mlp=kernel2)
        else:
            kernel2 = MLP([7 * embed_dim, 512, 256, hidden_channels[1]], paddle.nn.GELU)
            self.gno2 = NeighborMLPConvLayerLinear(mlp=kernel2)
        self.fno = FNO(
            fno_modes,
            hidden_channels=fno_hidden_channels,
            in_channels=hidden_channels[0] + 3 + in_channels,
            out_channels=fno_out_channels,
            use_mlp=True,
            mlp={"expansion": 1.0, "dropout": 0},
            domain_padding=fno_domain_padding,
            factorization=fno_factorization,
            norm=fno_norm,
            rank=fno_rank,
        )
        self.projection = Projection(
            in_channels=hidden_channels[1],
            out_channels=out_channels,
            hidden_channels=256,
            non_linearity=paddle.nn.functional.gelu,
            n_dim=1,
        )

    def forward(self, x_in, x_out, df, x_eval=None, area_in=None, area_eval=None):
        # manifold to latent neighborhood
        in_to_out_nb = self.nb_search_in(x_in, x_out.reshape((-1, 3)))

        # latent to manifold neighborhood
        if x_eval is not None:
            out_to_in_nb = self.nb_search_out(x_out.reshape((-1, 3)), x_eval)
        else:
            out_to_in_nb = self.nb_search_out(x_out.reshape((-1, 3)), x_in)

        # Embed manifold coordinates
        resolution = df.shape[-1]
        n_in = x_in.shape[0]
        if area_in is None or self.weighted_kernel is False:
            area_in = paddle.ones(shape=(n_in,))
        x_in = paddle.concat(x=[x_in, area_in.unsqueeze(axis=-1)], axis=-1)
        x_in_embed = self.pos_embed(x_in.reshape((-1,))).reshape((n_in, -1))

        if x_eval is not None:
            n_eval = x_eval.shape[0]
            if area_eval is None or self.weighted_kernel is False:
                area_eval = paddle.ones(shape=(n_eval,))
            x_eval = paddle.concat(x=[x_eval, area_eval.unsqueeze(axis=-1)], axis=-1)
            x_eval_embed = self.pos_embed(x_eval.reshape((-1,))).reshape((n_eval, -1))

        # Embed latent space coordinates
        x_out_embed = self.pos_embed(x_out.reshape((-1,))).reshape(
            (resolution**3, -1)
        )

        # Embed latent space features
        df_embed = self.df_embed(df.transpose(perm=[1, 2, 3, 0])).reshape(
            (resolution**3, -1)
        )
        grid_embed = paddle.concat(x=[x_out_embed, df_embed], axis=-1)

        # GNO : project to latent space
        u = self.gno1(x_in_embed, in_to_out_nb, grid_embed, area_in)
        u = (
            u.reshape((resolution, resolution, resolution, -1))
            .transpose(perm=[3, 0, 1, 2])
            .unsqueeze(axis=0)
        )

        # Add positional embedding and distance information
        u = paddle.concat(
            x=(
                x_out.transpose(perm=[3, 0, 1, 2]).unsqueeze(axis=0),
                df.unsqueeze(axis=0),
                u,
            ),
            axis=1,
        )
        # FNO on latent space
        u = self.fno(u)
        u = u.squeeze().transpose(perm=[1, 2, 3, 0]).reshape((resolution**3, -1))

        # GNO : project to manifold
        if not self.linear_kernel:
            if x_eval is not None:
                u = self.gno2(u, out_to_in_nb, x_eval_embed)
            else:
                u = self.gno2(u, out_to_in_nb, x_in_embed)
        elif x_eval is not None:
            u = self.gno2(
                x_in=x_out_embed,
                neighbors=out_to_in_nb,
                in_features=u,
                x_out=x_eval_embed,
            )
        else:
            u = self.gno2(
                x_in=x_out_embed,
                neighbors=out_to_in_nb,
                in_features=u,
                x_out=x_in_embed,
            )
        u = u.unsqueeze(axis=0).transpose(perm=[0, 2, 1])

        # Pointwise projection to out channels
        u = self.projection(u).transpose(perm=[0, 2, 1])
        return u

    @paddle.no_grad()
    def eval_dict(self, data_dict, loss_fn=None, decode_fn=None, **kwargs):
        x_in, x_out, df = self.data_dict_to_input(data_dict)
        pred = self(x_in, x_out, df)

        if loss_fn is None:
            loss_fn = self.loss

        truth = data_dict["pressure"].reshape((-1, 1))
        correct_truth_value = max(pred.shape) == max(truth.shape)
        out_dict = {}

        if correct_truth_value:
            out_dict["l2"] = loss_fn(pred, truth)
        if decode_fn is not None:
            pred = decode_fn(pred)
            truth = decode_fn(truth)
            out_dict["pred p"] = pred

            if correct_truth_value:
                out_dict["l2_decoded"] = loss_fn(pred, truth)

        return out_dict

    def loss_dict(self, data_dict, loss_fn=None):
        x_in, x_out, df = self.data_dict_to_input(data_dict)
        pred = self(x_in, x_out, df)
        if loss_fn is None:
            loss_fn = self.loss
        return {
            "loss": loss_fn(
                pred.reshape((1, -1)), data_dict["pressure"].reshape((1, -1))
            )
        }


class GNOFNOGNOAhmed(GNOFNOGNO):
    def __init__(
        self,
        radius_in=0.05,
        radius_out=0.05,
        embed_dim=16,
        hidden_channels=(86, 86),
        in_channels=2,
        out_channels=1,
        fno_modes=(32, 32, 32),
        fno_hidden_channels=86,
        fno_out_channels=86,
        fno_domain_padding=0.125,
        fno_norm="ada_in",
        adain_embed_dim=64,
        fno_factorization="tucker",
        fno_rank=0.4,
        linear_kernel=True,
        weighted_kernel=True,
        max_in_points=5000,
        subsample_train=1,
        subsample_eval=1,
    ):
        if fno_norm == "ada_in":
            init_norm = "group_norm"
        else:
            init_norm = fno_norm
        self.max_in_points = max_in_points
        self.subsample_train = subsample_train
        super().__init__(
            radius_in=radius_in,
            radius_out=radius_out,
            embed_dim=embed_dim,
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=sum(out_channels),
            fno_modes=fno_modes,
            fno_hidden_channels=fno_hidden_channels,
            fno_out_channels=fno_out_channels,
            fno_domain_padding=fno_domain_padding,
            fno_norm=init_norm,
            fno_factorization=fno_factorization,
            fno_rank=fno_rank,
            linear_kernel=linear_kernel,
            weighted_kernel=weighted_kernel,
        )
        if fno_norm == "ada_in":
            self.adain_pos_embed = PositionalEmbedding(adain_embed_dim)
            self.fno.fno_blocks.norm = paddle.nn.LayerList(
                sublayers=(
                    AdaIN(adain_embed_dim, fno_hidden_channels)
                    for _ in range(
                        self.fno.fno_blocks.n_norms * self.fno.fno_blocks.convs.n_layers
                    )
                )
            )
            self.use_adain = True
        else:
            self.use_adain = False

    def data_dict_to_input(self, data_dict):
        inlet_vel = 30.0
        x_in = data_dict[1][0]
        x_out = (
            data_dict[4][0].transpose(perm=[1, 2, 3, 0])
        )
        df = data_dict[2]
        area = data_dict[5][0]
        info_fields = inlet_vel * paddle.ones_like(x=df)
        df = paddle.concat(x=(df, info_fields), axis=0)
        if self.use_adain:
            vel = (
                paddle.to_tensor(data=[inlet_vel])
                .reshape((-1,))
                .astype(paddle.float32)
            )
            vel_embed = self.adain_pos_embed(vel)
            for norm in self.fno.fno_blocks.norm:
                norm.set_embedding(vel_embed)
        return x_in, x_out, df, area

    def forward(self, data_dict, device=None):
        x_in, x_out, df, area = self.data_dict_to_input(data_dict)
        # x_in [403016, 3], x_out [64, 64, 64, 3], df [2, 64, 64, 64], area [4000]
        if self.training:
            pred = super().forward(
                x_in,
                x_out,
                df,
                data_dict["centroids_sampled"][0],
                area,
                data_dict["areas_sampled"][0],
            )
        else:
            r = min(self.max_in_points, x_in.shape[0])

            def create_sections(length, r):
                sections = [r] * (length // r)
                if length % r != 0:
                    sections.append(-1)
                return sections

            pred_chunks = []
            x_in_sections = create_sections(x_in.shape[0], r)
            area_sections = create_sections(area.shape[0], r)
            x_in_chunks = paddle.split(x=x_in, num_or_sections=x_in_sections, axis=0)
            area_chunks = paddle.split(x=area, num_or_sections=area_sections, axis=0)
            for x_eval, area_eval in zip(x_in_chunks, area_chunks):
                pred_mini = super().forward(x_in, x_out, df, x_eval, area, area_eval)
                pred_chunks.append(pred_mini)
            pred = paddle.concat(x=tuple(pred_chunks), axis=1)
        return pred
