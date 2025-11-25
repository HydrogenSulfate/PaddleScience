import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[0]


def run(cmd, cwd):
    """Run shell commands with subprocess, raising on failure."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "3"
    subprocess.run(cmd, cwd=cwd, shell=True, check=True, env=env)


@pytest.fixture(scope="function")
def workdir(tmp_path):
    """A helper to emulate pushd/popd behavior."""
    old = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(old)


# ------------------------------------------------------------
# Global setup once for all tests
# ------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def global_setup():
    os.environ["PYTHONPATH"] = str(ROOT)
    os.environ["MAX_ITERS"] = "3"
    os.environ.pop("https_proxy", None)
    os.environ.pop("http_proxy", None)

    run("python -m pip install --upgrade pip", cwd=ROOT)
    run("python -m pip install uv", cwd=ROOT)


# ============================================================
#                 INDIVIDUAL EXAMPLE TESTS
# ============================================================


def test_deephpms():
    d = ROOT / "examples/deephpms"
    run(
        "wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/DeepHPMs/burgers_sine.mat -P ./datasets/",
        cwd=d,
    )
    run(
        "python burgers.py "
        "DATASET_PATH=./datasets/burgers_sine.mat "
        "DATASET_PATH_SOL=./datasets/burgers_sine.mat",
        cwd=d,
    )


def test_deeponet_operator_learning():
    d = ROOT / "examples/operator_learning"
    run(
        "wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/DeepONet/antiderivative_unaligned_train.npz",
        cwd=d,
    )
    run(
        "wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/DeepONet/antiderivative_unaligned_test.npz",
        cwd=d,
    )
    run("python deeponet.py", cwd=d)


def test_euler_beam():
    run("python euler_beam.py", ROOT / "examples/euler_beam")


def test_laplace2d():
    run("python laplace2d.py", ROOT / "examples/laplace")


def test_lorenz():
    d = ROOT / "examples/lorenz"
    run(
        "wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/lorenz_training_rk.hdf5 -P ./datasets/",
        cwd=d,
    )
    run(
        "wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/lorenz_valid_rk.hdf5 -P ./datasets/",
        cwd=d,
    )
    run("python train_enn.py", cwd=d)
    # run("python train_transformer.py", cwd=d)


def test_pirbn():
    run("python main.py", ROOT / "jointContribution/PIRBN")


def test_rossler():
    d = ROOT / "examples/rossler"
    run(
        "wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/rossler_training.hdf5 -P ./datasets/",
        cwd=d,
    )
    run(
        "wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/rossler_valid.hdf5 -P ./datasets/",
        cwd=d,
    )
    run("python train_enn.py", cwd=d)
    # run("python train_transformer.py", cwd=d)


def test_ide():
    run("python volterra_ide.py", ROOT / "examples/ide")


def test_amgnet():
    d = ROOT / "examples/amgnet"
    run(
        "wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/AMGNet/data.zip",
        cwd=d,
    )
    run("unzip -o data.zip", cwd=d)
    run("python amgnet_airfoil.py", cwd=d)


def test_aneurysm():
    d = ROOT / "examples/aneurysm"
    run(
        "wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/aneurysm/aneurysm_dataset.tar",
        cwd=d,
    )
    run("tar -xvf aneurysm_dataset.tar", cwd=d)
    run("python aneurysm.py", cwd=d)


def test_bubble():
    d = ROOT / "examples/bubble"
    run(
        "wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/BubbleNet/bubble.mat",
        cwd=d,
    )
    run("python bubble.py", cwd=d)


def test_cylinder2d_unsteady():
    d = ROOT / "examples/cylinder/2d_unsteady"
    run(
        "wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/cylinder2d_unsteady_Re100/cylinder2d_unsteady_Re100_dataset.tar",
        cwd=d,
    )
    run("tar -xvf cylinder2d_unsteady_Re100_dataset.tar", cwd=d)
    run("python cylinder2d_unsteady_Re100.py", cwd=d)


def test_cylinder2d_transformer_physx():
    d = ROOT / "examples/cylinder/2d_unsteady/transformer_physx"
    run(
        "wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/cylinder_training.hdf5 -P ./datasets/",
        cwd=d,
    )
    run(
        "wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/cylinder_valid.hdf5 -P ./datasets/",
        cwd=d,
    )
    run("python train_enn.py", cwd=d)
    # run("python train_transformer.py", cwd=d)


def test_darcy2d():
    run("python darcy2d.py", ROOT / "examples/darcy")


def test_deepcfd():
    d = ROOT / "examples/deepcfd"
    run(
        "wget -nc -P ./datasets/ https://paddle-org.bj.bcebos.com/paddlescience/datasets/DeepCFD/dataX.pkl",
        cwd=d,
    )
    run(
        "wget -nc -P ./datasets/ https://paddle-org.bj.bcebos.com/paddlescience/datasets/DeepCFD/dataY.pkl",
        cwd=d,
    )
    run("python deepcfd.py", cwd=d)


def test_nsfnet():
    run("python VP_NSFNet1.py", ROOT / "examples/nsfnet")


def test_fsi_viv():
    run("python viv.py", ROOT / "examples/fsi")


def test_biharmonic2d():
    run("python biharmonic2d.py", ROOT / "examples/biharmonic2d")


def test_bracket():
    d = ROOT / "examples/bracket"
    run(
        "wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/bracket/bracket_dataset.tar",
        cwd=d,
    )
    run("tar -xvf bracket_dataset.tar", cwd=d)
    run("python bracket.py", cwd=d)


def test_control_arm():
    d = ROOT / "examples/control_arm"
    run(
        "wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/control_arm/control_arm.stl -P ./datasets/",
        cwd=d,
    )
    run("python forward_analysis.py", cwd=d)


def test_epnn():
    d = ROOT / "examples/epnn"
    run(
        "wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/epnn/dstate-16-plas.dat -P ./datasets/",
        cwd=d,
    )
    run(
        "wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/epnn/dstress-16-plas.dat -P ./datasets/",
        cwd=d,
    )
    run("python epnn.py", cwd=d)


def test_phylstm():
    d = ROOT / "examples/phylstm"
    run(
        "wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyLSTM/data_boucwen.mat",
        cwd=d,
    )
    run("python phylstm2.py", cwd=d)


def test_topopt():
    d = ROOT / "examples/topopt"
    run(
        "wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/topopt/top_dataset.h5 -P ./datasets/",
        cwd=d,
    )
    run("python topopt.py", cwd=d)


def test_heat_exchanger():
    run("python heat_exchanger.py", ROOT / "examples/heat_exchanger")


def test_heat_pinn():
    run("python heat_pinn.py", ROOT / "examples/heat_pinn")


def test_phygeonet():
    d = ROOT / "examples/phygeonet"
    run(
        "wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyGeoNet/heat_equation.npz -P ./data/",
        cwd=d,
    )
    run("python heat_equation.py", cwd=d)


def test_chip_heat():
    run("python chip_heat.py", ROOT / "examples/chip_heat")
