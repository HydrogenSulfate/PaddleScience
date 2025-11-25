import os
import subprocess

# import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent  # 根目录


def run_cmd(cmd, cwd):
    """在指定目录执行命令"""
    print(f"\n=== Running: {cmd} in {cwd} ===\n")
    env = os.environ.copy()
    env["FLAGS_enable_api_kernel_fallback"] = "0"
    # env["FLAGS_call_stack_level"] = "3"
    env["CUDA_VISIBLE_DEVICES"] = "11"
    result = subprocess.run(
        cmd,
        cwd=cwd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        env=env,
    )
    print(result.stdout)
    assert result.returncode == 0, f"Command failed: {cmd}\n{result.stdout}"


"""
PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 python -u -m pytest -s --capture=no -vvv -x test_example.py --timeout=300 --lf
"""


@pytest.mark.parametrize(
    "cwd, cmd",
    [
        # ("examples/allen_cahn", "python allen_cahn_piratenet.py"),
        (
            "examples/deephpms",
            "python burgers.py DATASET_PATH=./datasets/burgers_sine.mat DATASET_PATH_SOL=./datasets/burgers_sine.mat",
        ),
        ("examples/operator_learning", "python deeponet.py"),
        ("examples/euler_beam", "python euler_beam.py"),
        ("examples/laplace", "python laplace2d.py"),
        ("examples/lorenz", "python train_transformer.py"),
        ("jointContribution/PIRBN", "python main.py"),
        ("examples/rossler", "python train_transformer.py"),
        ("examples/ide", "python volterra_ide.py"),
        # ("examples/NLS-MB", "python NLS-MB_optical_soliton.py"),
        # ("examples/spinn", "python helmholtz3d.py"),
        # ("examples/xpinn", "python xpinn.py"),
        # ("examples/neuraloperator", "python train_tfno.py"),
        # ("examples/brusselator3d", "python brusselator3d.py"),
        # ("examples/transformer4sr", "python transformer4sr.py"),
        # ("examples/LatentNO", "python LatentNO-steady.py --config-name=LatentNO-Darcy.yaml"),
        # ("examples/fundiff", "python main.py -cn fae.yaml"),
        # ("examples/catheter", "python catheter.py"),
        ("examples/amgnet", "python amgnet_airfoil.py"),
        ("examples/aneurysm", "python aneurysm.py"),
        ("examples/bubble", "python bubble.py"),
        # ("examples/adv", "python adv_cvit.py"),
        ("examples/cylinder/2d_unsteady", "python cylinder2d_unsteady_Re100.py"),
        (
            "examples/cylinder/2d_unsteady/transformer_physx",
            "python train_transformer.py",
        ),
        ("examples/darcy", "python darcy2d.py"),
        ("examples/deepcfd", "python deepcfd.py"),
        # ("examples/drivaernet", "python drivaernet.py"),
        # ("examples/drivaernetplusplus", "python drivaernetplusplus.py"),
        # ("examples/ldc", "python ldc_2d_Re3200_sota.py"),
        # ("examples/ldc", "python ldc2d_unsteady_Re10.py"),
        # ("examples/aneurysm", "python aneurysm_flow.py"),
        ("examples/nsfnet", "python VP_NSFNet1.py"),
        # ("examples/phycrnet", "python main.py DATA_PATH=./data/burgers_1501x2x128x128.mat"),
        # ("examples/shock_wave", "python shock_wave.py"),
        # ("examples/tempoGAN", "python tempoGAN.py"),
        # ("examples/nsfnet", "python VP_NSFNet4.py mode=eval data_dir=./data/ EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/nsfnet/nsfnet4.pdparams"),
        ("examples/fsi", "python viv.py"),
        ("examples/biharmonic2d", "python biharmonic2d.py"),
        ("examples/bracket", "python bracket.py"),
        ("examples/control_arm", "python forward_analysis.py"),
        ("examples/epnn", "python epnn.py"),
        ("examples/phylstm", "python phylstm2.py"),
        ("examples/topopt", "python topopt.py"),
        # ("examples/ntopo", "python ntopo.py"),
        # ("examples/heart", "python inverse.py TRAIN.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/heart/inverse_pretrained.pdparams"),
        ("examples/heat_exchanger", "python heat_exchanger.py"),
        ("examples/heat_pinn", "python heat_pinn.py"),
        ("examples/phygeonet", "python heat_equation.py"),
        ("examples/chip_heat", "python chip_heat.py"),
        ("examples/hpinns", "python holography.py"),
        # ("examples/perovskite_solar_cells", "python psc_nn.py mode=train"),
        # ("examples/graphcast", "python graphcast.py mode=eval EVAL.pretrained_model_path=\"data/params/GraphCast_small---ERA5-1979-2015---resolution-1.0---pressure-levels-13---mesh-2to5---precipitation-input-and-output.pdparams\""),
        # ("examples/tgcn", "python run.py data_name=PEMSD8 mode=eval EVAL.pretrained_model_path=PEMSD8_pretrained_model.pdparams"),
        # ("examples/unetformer", "python vaihingen_test.py -c config/vaihingen/unetformer.py -o fig_results/vaihingen/unetformer --rgb"),
        # ("examples/smc_reac", "python smc_reac.py"),
        # ("examples/ifm", "python ifm.py mode=train data_label=tox21 MODEL.embed_name='IFM'"),
        # ("examples/synthemol", "python main.py mode=train"),
        ("examples/tadf/TADF_Est", "python Est.py mode=train"),
        # ("examples/stafnet", "python stafnet.py mode=eval EVAL.pretrained_model_path=\"https://paddle-org.bj.bcebos.com/paddlescience/models/stafnet/stafnet.pdparams\""),
        # 报错符合预期的
        # ("examples/UTAE", "python test_semantic.py --weight_file ./pretrained/semantic.pdparams --dataset_folder \"./data/PASTIS\" --device gpu --num_workers 0"),
        # ("examples/fengwu", "python predict.py INFER.device=cpu"),
        # ("examples/fuxi", "python predict.py"),
        # ("examples/pangu_weather", "python predict.py INFER.export_path=inference/pangu_weather_1 INFER.device=cpu"),
        # ("examples/pangu_weather", "python predict.py INFER.export_path=inference/pangu_weather_3 INFER.device=cpu"),
        # ("examples/pangu_weather", "python predict.py INFER.export_path=inference/pangu_weather_6 INFER.device=cpu"),
        # ("examples/pangu_weather", "python predict.py INFER.export_path=inference/pangu_weather_24 INFER.device=cpu"),
        # ("examples/velocityGAN", "python velocityGAN.py"),
        # ("examples/cgcnn", "python CGCNN.py"),
        # ("examples/nowcastnet", "python nowcastnet.py mode=infer"),
        # ("examples/dgmr", "python dgmr.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/dgmr/dgmr_pretrained.pdparams"),
    ],
)
def test_example(cwd, cmd):
    run_cmd(cmd, ROOT / cwd)
