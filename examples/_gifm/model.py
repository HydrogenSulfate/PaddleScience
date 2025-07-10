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



class Encoder(nn.Layer):
    def __init__(self):
        super().__init__()
        pass




class Decoder(nn.Layer):
    def __init__(self):
        super().__init__()

