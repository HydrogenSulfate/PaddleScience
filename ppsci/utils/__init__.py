# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import List

if TYPE_CHECKING:
    from ppsci.utils import ema
    from ppsci.utils import initializer
    from ppsci.utils import logger
    from ppsci.utils import misc
    from ppsci.utils import reader
    from ppsci.utils import writer
    from ppsci.utils.checker import dynamic_import_to_globals
    from ppsci.utils.checker import run_check
    from ppsci.utils.checker import run_check_mesh
    from ppsci.utils.expression import ExpressionSolver
    from ppsci.utils.misc import AverageMeter
    from ppsci.utils.misc import set_random_seed
    from ppsci.utils.reader import load_csv_file
    from ppsci.utils.reader import load_mat_file
    from ppsci.utils.reader import load_npz_file
    from ppsci.utils.reader import load_vtk_file
    from ppsci.utils.reader import load_vtk_with_time_file
    from ppsci.utils.save_load import load_checkpoint
    from ppsci.utils.save_load import load_pretrain
    from ppsci.utils.save_load import save_checkpoint
    from ppsci.utils.symbolic import lambdify
    from ppsci.utils.writer import save_csv_file
    from ppsci.utils.writer import save_tecplot_file

__all__ = [
    "AverageMeter",
    "ExpressionSolver",
    "initializer",
    "logger",
    "misc",
    "ema",
    "reader",
    "writer",
    "load_csv_file",
    "load_mat_file",
    "load_npz_file",
    "load_vtk_file",
    "load_vtk_with_time_file",
    "save_csv_file",
    "save_tecplot_file",
    "dynamic_import_to_globals",
    "run_check",
    "run_check_mesh",
    "set_random_seed",
    "load_checkpoint",
    "load_pretrain",
    "save_checkpoint",
    "lambdify",
]

# These are importable submodules. Only names in ``__all__`` are re-exported
# by ``from ppsci.utils import *``; the mapping below preserves attribute-style
# access without eagerly importing optional or heavyweight modules.
_SUBMODULES: Dict[str, str] = {
    "callbacks": "ppsci.utils.callbacks",
    "checker": "ppsci.utils.checker",
    "config": "ppsci.utils.config",
    "download": "ppsci.utils.download",
    "ema": "ppsci.utils.ema",
    "expression": "ppsci.utils.expression",
    "initializer": "ppsci.utils.initializer",
    "logger": "ppsci.utils.logger",
    "misc": "ppsci.utils.misc",
    "reader": "ppsci.utils.reader",
    "registry": "ppsci.utils.registry",
    "save_load": "ppsci.utils.save_load",
    "symbolic": "ppsci.utils.symbolic",
    "writer": "ppsci.utils.writer",
}

_ATTRS: Dict[str, tuple] = {
    "AverageMeter": ("ppsci.utils.misc", "AverageMeter"),
    "ExpressionSolver": ("ppsci.utils.expression", "ExpressionSolver"),
    "dynamic_import_to_globals": ("ppsci.utils.checker", "dynamic_import_to_globals"),
    "lambdify": ("ppsci.utils.symbolic", "lambdify"),
    "load_checkpoint": ("ppsci.utils.save_load", "load_checkpoint"),
    "load_csv_file": ("ppsci.utils.reader", "load_csv_file"),
    "load_mat_file": ("ppsci.utils.reader", "load_mat_file"),
    "load_npz_file": ("ppsci.utils.reader", "load_npz_file"),
    "load_pretrain": ("ppsci.utils.save_load", "load_pretrain"),
    "load_vtk_file": ("ppsci.utils.reader", "load_vtk_file"),
    "load_vtk_with_time_file": ("ppsci.utils.reader", "load_vtk_with_time_file"),
    "run_check": ("ppsci.utils.checker", "run_check"),
    "run_check_mesh": ("ppsci.utils.checker", "run_check_mesh"),
    "save_checkpoint": ("ppsci.utils.save_load", "save_checkpoint"),
    "save_csv_file": ("ppsci.utils.writer", "save_csv_file"),
    "save_tecplot_file": ("ppsci.utils.writer", "save_tecplot_file"),
    "set_random_seed": ("ppsci.utils.misc", "set_random_seed"),
}


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        value = importlib.import_module(_SUBMODULES[name])
    elif name in _ATTRS:
        module_name, attr_name = _ATTRS[name]
        module = importlib.import_module(module_name)
        value = getattr(module, attr_name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    globals()[name] = value
    return value


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(__all__))
