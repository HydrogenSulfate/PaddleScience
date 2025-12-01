import os
import subprocess

import pytest

# ==========================================
# 1. Environment Setup (Global Fixture)
# ==========================================


@pytest.fixture(scope="session")
def global_env():
    """
    Equivalent to the initial export commands in the shell script.
    """
    env = os.environ.copy()

    # Set explicit environment variables from the script
    env.pop("PADDLE_ELASTIC_JOB_ID", None)
    env.pop("PADDLE_TRAINER_ENDPOINTS", None)
    env.pop("DISTRIBUTED_TRAINER_ENDPOINTS", None)
    env.pop("FLAGS_START_PORT", None)
    env.pop("PADDLE_ELASTIC_TIMEOUT", None)
    env["NNODES"] = "1"
    env["PADDLE_TRAINERS_NUM"] = "1"

    # Handle PYTHONPATH: append current working directory
    current_cwd = os.getcwd()
    original_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{current_cwd}:{original_pythonpath}"

    env["MAX_ITERS"] = "3"
    env["HYDRA_FULL_ERROR"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "1"

    # Unset proxies
    # env.pop("https_proxy", None)
    # env.pop("http_proxy", None)
    return env


def run_cmds(commands, cwd, env):
    """
    Helper function to run a list of shell commands in a specific directory.
    """
    # Ensure directory exists (some scripts might assume it, though usually they exist in repo)
    if not os.path.exists(cwd):
        print(f"Warning: Directory {cwd} does not exist, attempting to run anyway...")

    for cmd in commands:
        print(f"\n[Running] {cmd} in {cwd}")
        # shell=True is used to support pipes like '| tee' and wildcards
        result = subprocess.run(cmd, shell=True, cwd=cwd, env=env, text=True)
        if result.returncode != 0:
            pytest.fail(f"Command failed with return code {result.returncode}: {cmd}")


# ==========================================
# 2. Test Cases
# ==========================================

# --- zh/examples/allen_cahn.md ---
@pytest.mark.skip(reason="Commented out in original script")
def test_allen_cahn(global_env):
    commands = [
        "python allen_cahn_piratenet.py 2>&1 | tee allen_cahn.log",
    ]
    run_cmds(commands, cwd="examples/allen_cahn/", env=global_env)


# --- zh/examples/deephpms.md ---
def test_deephpms(global_env):
    commands = [
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/DeepHPMs/burgers_sine.mat -P ./datasets/",
        "python burgers.py DATASET_PATH=./datasets/burgers_sine.mat DATASET_PATH_SOL=./datasets/burgers_sine.mat 2>&1 | tee deephpms.log",
    ]
    run_cmds(commands, cwd="examples/deephpms/", env=global_env)


# --- zh/examples/deeponet.md ---
def test_deeponet(global_env):
    commands = [
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/DeepONet/antiderivative_unaligned_train.npz",
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/DeepONet/antiderivative_unaligned_test.npz",
        "python deeponet.py 2>&1 | tee operator_learnin.log",
    ]
    run_cmds(commands, cwd="examples/operator_learning/", env=global_env)


# --- zh/examples/euler_beam.md ---
def test_euler_beam(global_env):
    commands = [
        "python euler_beam.py 2>&1 | tee euler_beam.log",
    ]
    run_cmds(commands, cwd="examples/euler_beam/", env=global_env)


# --- zh/examples/laplace2d.md ---
def test_laplace2d(global_env):
    commands = [
        "python laplace2d.py 2>&1 | tee laplace.log",
    ]
    run_cmds(commands, cwd="examples/laplace/", env=global_env)


# --- zh/examples/lorenz.md ---
def test_lorenz(global_env):
    commands = [
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/lorenz_training_rk.hdf5 -P ./datasets/",
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/lorenz_valid_rk.hdf5 -P ./datasets/",
        "python train_enn.py 2>&1 | tee lorenz.log",
        # "python train_transformer.py" # This was commented in original,
    ]
    run_cmds(commands, cwd="examples/lorenz/", env=global_env)


# --- zh/examples/pirbn.md ---
def test_pirbn(global_env):
    commands = [
        "python main.py 2>&1 | tee PIRBN.log",
    ]
    run_cmds(commands, cwd="jointContribution/PIRBN", env=global_env)


# --- zh/examples/rossler.md ---
def test_rossler(global_env):
    commands = [
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/rossler_training.hdf5 -P ./datasets/ 2>&1 | tee rossler.log",
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/rossler_valid.hdf5 -P ./datasets/",
        "python train_enn.py",
        # "python train_transformer.py" # Commented in original,
    ]
    run_cmds(commands, cwd="examples/rossler/", env=global_env)


# --- zh/examples/volterra_ide.md ---
def test_volterra_ide(global_env):
    commands = [
        "python volterra_ide.py 2>&1 | tee ide.log",
    ]
    run_cmds(commands, cwd="examples/ide/", env=global_env)


# --- zh/examples/nlsmb.md ---
@pytest.mark.skip(reason="Commented out in original script")
def test_nlsmb(global_env):
    commands = [
        "python NLS-MB_optical_rogue_wave.py 2>&1 | tee MB.log",
        "python NLS-MB_optical_soliton.py",
    ]
    run_cmds(commands, cwd="examples/NLS-MB/", env=global_env)


# --- zh/examples/spinn.md ---
@pytest.mark.skip(reason="Commented out in original script")
def test_spinn(global_env):
    commands = [
        "python helmholtz3d.py 2>&1 | tee spinn.log",
    ]
    run_cmds(commands, cwd="examples/spinn/", env=global_env)


# --- zh/examples/xpinns.md ---
# @pytest.mark.skip(reason="Commented out in original script")
def test_xpinns(global_env):
    commands = [
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/XPINN/XPINN_2D_PoissonEqn.mat -P ./data/ 2>&1 | tee xpinn.log",
        "python xpinn.py",
    ]
    run_cmds(commands, cwd="examples/xpinn/", env=global_env)


# --- zh/examples/neuraloperator.md ---
@pytest.mark.skip(reason="Commented out in original script")
def test_neuraloperator(global_env):
    commands = [
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/neuraloperator/darcy_flow/darcy_train_16.npy -P ./datasets/darcyflow/",
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/neuraloperator/darcy_flow/darcy_test_32.npy -P ./datasets/darcyflow/",
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/neuraloperator/darcy_flow/darcy_test_16.npy -P ./datasets/darcyflow/",
        "python train_tfno.py 2>&1 | tee neuraloperator.log",
        "python train_uno.py",
    ]
    run_cmds(commands, cwd="examples/neuraloperator/", env=global_env)


# --- zh/examples/brusselator3d.md ---
@pytest.mark.skip(reason="Commented out in original script")
def test_brusselator3d(global_env):
    commands = [
        "python brusselator3d.py 2>&1 | tee brusselator3d.log",
    ]
    run_cmds(commands, cwd="examples/brusselator3d/", env=global_env)


# --- zh/examples/transformer4sr.md ---
@pytest.mark.skip(reason="Commented out in original script")
def test_transformer4sr(global_env):
    commands = [
        "python -m pip install zss -i https://pypi.tuna.tsinghua.edu.cn/simple",
        "tar -xzvf data_generated.tar.gz",
        "python transformer4sr.py 2>&1 | tee transformer4s.log",
    ]
    run_cmds(commands, cwd="examples/transformer4sr/", env=global_env)


# --- zh/examples/latent_no.md ---
@pytest.mark.skip(reason="Commented out in original script")
def test_latent_no(global_env):
    commands = [
        "python LatentNO-steady.py --config-name=LatentNO-Darcy.yaml 2>&1 | tee LatentNO.log",
    ]
    run_cmds(commands, cwd="examples/LatentNO/", env=global_env)


# --- zh/examples/fundiff.md ---
@pytest.mark.skip(reason="Commented out in original script")
def test_fundiff(global_env):
    commands = [
        "python main.py -cn fae.yaml 2>&1 | tee fundiff.log",
    ]
    run_cmds(commands, cwd="examples/fundiff/", env=global_env)


# --- zh/examples/catheter.md ---
@pytest.mark.skip(reason="Commented out in original script")
def test_catheter(global_env):
    commands = [
        "wget -c https://dataset.bj.bcebos.com/PaddleScience/2024%20AI-aided%20geometric%20design%20of%20anti-infection%20catheters/data.zip",
        "unzip -o data.zip",
        "python catheter.py 2>&1 | tee catheter.log",
    ]
    run_cmds(commands, cwd="examples/catheter/", env=global_env)


# --- zh/examples/amgnet.md ---
@pytest.mark.skip(reason="Commented out in original script")
def test_amgnet(global_env):
    commands = [
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/AMGNet/data.zip",
        "unzip -o data.zip",
        "python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple",
        "python amgnet_airfoil.py 2>&1 | tee amgnet.log",
    ]
    run_cmds(commands, cwd="examples/amgnet/", env=global_env)


# --- zh/examples/aneurysm.md ---
def test_aneurysm(global_env):
    commands = [
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/aneurysm/aneurysm_dataset.tar",
        "tar -xvf aneurysm_dataset.tar",
        "python aneurysm.py 2>&1 | tee aneurysm.log",
    ]
    run_cmds(commands, cwd="examples/aneurysm/", env=global_env)


# --- zh/examples/bubble.md ---
def test_bubble(global_env):
    commands = [
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/BubbleNet/bubble.mat",
        "python bubble.py 2>&1 | tee bubble.log",
    ]
    run_cmds(commands, cwd="examples/bubble/", env=global_env)


# --- zh/examples/cfdgcn.md ---
@pytest.mark.skip(reason="Commented out in original script")
def test_cfdgcn(global_env):
    # Note: path in script was examples/allen_cahn/, assuming copy paste error in original script comment?
    # Keeping as per original script logic
    commands = [
        "python allen_cahn_piratenet.py 2>&1 | tee allen_cahn.log",
    ]
    run_cmds(commands, cwd="examples/allen_cahn/", env=global_env)


# --- zh/examples/adv_cvit.md ---
@pytest.mark.skip(reason="Commented out in original script")
def test_adv_cvit(global_env):
    commands = [
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/Cvit_adv/adv_a0.npy -P ./data",
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/Cvit_adv/adv_aT.npy -P ./data",
        "python adv_cvit.py 2>&1 | tee adv.log",
    ]
    run_cmds(commands, cwd="examples/adv/", env=global_env)


# --- zh/examples/ns_cvit.md ---
@pytest.mark.skip(reason="Commented out in original script")
def test_ns_cvit(global_env):
    # Assuming this reuses allen_cahn as per input script
    commands = [
        "python allen_cahn_piratenet.py 2>&1 | tee allen_cahn.log",
    ]
    run_cmds(commands, cwd="examples/allen_cahn/", env=global_env)


# --- zh/examples/cylinder2d_unsteady.md ---
def test_cylinder2d_unsteady(global_env):
    commands = [
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/cylinder2d_unsteady_Re100/cylinder2d_unsteady_Re100_dataset.tar",
        "tar -xvf cylinder2d_unsteady_Re100_dataset.tar",
        "python cylinder2d_unsteady_Re100.py 2>&1 | tee 2d_unsteady.log",
    ]
    run_cmds(commands, cwd="examples/cylinder/2d_unsteady/", env=global_env)


# --- zh/examples/cylinder2d_unsteady_transformer_physx.md ---
def test_cylinder2d_transformer(global_env):
    commands = [
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/cylinder_training.hdf5 -P ./datasets/",
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/cylinder_valid.hdf5 -P ./datasets/",
        "python train_enn.py 2>&1 | tee transformer_phys.log"
        # "python train_transformer.py" # Commented,
    ]
    run_cmds(
        commands, cwd="examples/cylinder/2d_unsteady/transformer_physx", env=global_env
    )


# --- zh/examples/darcy2d.md ---
def test_darcy2d(global_env):
    commands = [
        "python darcy2d.py 2>&1 | tee darcy.log",
    ]
    run_cmds(commands, cwd="examples/darcy/", env=global_env)


# --- zh/examples/deepcfd.md ---
def test_deepcfd(global_env):
    commands = [
        "wget -c -P ./datasets/ https://paddle-org.bj.bcebos.com/paddlescience/datasets/DeepCFD/dataX.pkl",
        "wget -c -P ./datasets/ https://paddle-org.bj.bcebos.com/paddlescience/datasets/DeepCFD/dataY.pkl",
        "python deepcfd.py 2>&1 | tee deepcfd.log",
    ]
    run_cmds(commands, cwd="examples/deepcfd/", env=global_env)


# --- zh/examples/drivaernet.md ---
@pytest.mark.skip(reason="Commented out in original script")
def test_drivaernet(global_env):
    commands = [
        "wget -c https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAer%2B%2B/data.tar",
        "tar -xvf data.tar",
        "python drivaernet.py 2>&1 | tee drivaernet.log",
    ]
    run_cmds(commands, cwd="examples/drivaernet/", env=global_env)


# --- zh/examples/drivaernetplusplus.md ---
@pytest.mark.skip(reason="Commented out in original script")
def test_drivaernetplusplus(global_env):
    commands = [
        "mkdir -p data/subset_dir",
        "wget -c https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAer%2B%2B/DrivAer%2B%2B_Points.tar",
        "tar -xvf DrivAer++_Points.tar -C ./data",
        "wget -c https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAer%2B%2B/DrivAerNetPlusPlus_Drag_8k.csv -P ./data",
        "wget -c https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAer%2B%2B/test_design_ids.txt -P ./data/subset_dir",
        "wget -c https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAer%2B%2B/train_design_ids.txt -P ./data/subset_dir",
        "wget -c https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAer%2B%2B/val_design_ids.txt -P ./data/subset_dir",
        "mv ./data/workspace/gino_data/14_DrivAer++/paddle_tensor ./data/DrivAerNetPlusPlus_Processed_Point_Clouds_100k_paddle",
        "rm -rf data/workspace",
        "python drivaernetplusplus.py 2>&1 | tee drivaernetplusplus.log",
    ]
    run_cmds(commands, cwd="examples/drivaernetplusplus/", env=global_env)


# --- zh/examples/ldc2d_steady.md ---
@pytest.mark.skip(reason="Commented out in original script")
def test_ldc2d_steady(global_env):
    commands = [
        "wget -c -P ./data/ https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re100.mat https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re400.mat https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re1000.mat https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re3200.mat",
        "python ldc_2d_Re3200_sota.py 2>&1 | tee ldc.log",
    ]
    run_cmds(commands, cwd="examples/ldc/", env=global_env)


# --- zh/examples/ldc2d_unsteady.md ---
@pytest.mark.skip(reason="Commented out in original script")
def test_ldc2d_unsteady(global_env):
    commands = [
        "python ldc2d_unsteady_Re10.py 2>&1 | tee ldc.log",
    ]
    run_cmds(commands, cwd="examples/ldc/", env=global_env)


# --- zh/examples/labelfree_DNN_surrogate.md ---
@pytest.mark.skip(reason="Commented out in original script")
def test_labelfree_dnn(global_env):
    commands = [
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/LabelFree-DNN-Surrogate/LabelFree-DNN-Surrogate_data.zip",
        "unzip -o LabelFree-DNN-Surrogate_data.zip",
        "python aneurysm_flow.py 2>&1 | tee aneurysm_flow.log",
    ]
    run_cmds(commands, cwd="examples/aneurysm/", env=global_env)


# --- zh/examples/nsfnet.md ---
def test_nsfnet(global_env):
    commands = [
        "python VP_NSFNet1.py 2>&1 | tee nsfnet.log",
    ]
    run_cmds(commands, cwd="examples/nsfnet/", env=global_env)


# --- zh/examples/phycrnet.md ---
@pytest.mark.skip(reason="Commented out in original script")
def test_phycrnet(global_env):
    commands = [
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyCRNet/burgers_1501x2x128x128.mat -P ./data/",
        "python main.py DATA_PATH=./data/burgers_1501x2x128x128.mat 2>&1 | tee phycrnet.log",
    ]
    run_cmds(commands, cwd="examples/phycrnet/", env=global_env)


# --- zh/examples/shock_wave.md ---
@pytest.mark.skip(reason="Commented out in original script")
def test_shock_wave(global_env):
    commands = [
        "python shock_wave.py 2>&1 | tee shock_wave.log",
    ]
    run_cmds(commands, cwd="examples/shock_wave/", env=global_env)


# --- zh/examples/tempoGAN.md ---
@pytest.mark.skip(reason="Commented out in original script")
def test_tempogan(global_env):
    commands = [
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/tempoGAN/2d_train.mat -P datasets/tempoGAN/",
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/tempoGAN/2d_valid.mat -P datasets/tempoGAN/",
        "python tempoGAN.py 2>&1 | tee tempoGAN.log",
    ]
    run_cmds(commands, cwd="examples/tempoGAN/", env=global_env)


# --- zh/examples/nsfnet4.md ---
@pytest.mark.skip(reason="Commented out in original script")
def test_nsfnet4(global_env):
    commands = [
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/NSFNet/NSF4_data.zip -P ./data/",
        "unzip -o ./data/NSF4_data.zip",
        "python VP_NSFNet4.py mode=eval data_dir=./data/ EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/nsfnet/nsfnet4.pdparams 2>&1 | tee nsfnet.log",
    ]
    run_cmds(commands, cwd="examples/nsfnet/", env=global_env)


# --- zh/examples/viv.md ---
def test_viv(global_env):
    commands = [
        "python viv.py 2>&1 | tee fsi.log",
    ]
    run_cmds(commands, cwd="examples/fsi/", env=global_env)


# --- zh/examples/biharmonic2d.md ---
def test_biharmonic2d(global_env):
    commands = [
        "python biharmonic2d.py 2>&1 | tee biharmonic2d.log",
    ]
    run_cmds(commands, cwd="examples/biharmonic2d/", env=global_env)


# --- zh/examples/bracket.md ---
def test_bracket(global_env):
    commands = [
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/bracket/bracket_dataset.tar",
        "tar -xvf bracket_dataset.tar",
        "python bracket.py 2>&1 | tee bracket.log",
    ]
    run_cmds(commands, cwd="examples/bracket/", env=global_env)


# --- zh/examples/control_arm.md ---
def test_control_arm(global_env):
    commands = [
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/control_arm/control_arm.stl -P ./datasets/",
        "python forward_analysis.py 2>&1 | tee control_arm.log",
    ]
    run_cmds(commands, cwd="examples/control_arm/", env=global_env)


# --- zh/examples/epnn.md ---
def test_epnn(global_env):
    commands = [
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/epnn/dstate-16-plas.dat -P ./datasets/",
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/epnn/dstress-16-plas.dat -P ./datasets/",
        "python epnn.py 2>&1 | tee epnn.log",
    ]
    run_cmds(commands, cwd="examples/epnn/", env=global_env)


# --- zh/examples/phylstm.md ---
def test_phylstm(global_env):
    commands = [
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyLSTM/data_boucwen.mat",
        "python phylstm2.py 2>&1 | tee phylstm.log",
    ]
    run_cmds(commands, cwd="examples/phylstm/", env=global_env)


# --- zh/examples/topopt.md ---
def test_topopt(global_env):
    commands = [
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/topopt/top_dataset.h5 -P ./datasets/",
        "python topopt.py 2>&1 | tee topopt.log",
    ]
    run_cmds(commands, cwd="examples/topopt/", env=global_env)


# --- zh/examples/ntopo.md ---
@pytest.mark.skip(reason="Commented out in original script")
def test_ntopo(global_env):
    commands = [
        "python ntopo.py 2>&1 | tee ntop.log",
    ]
    run_cmds(commands, cwd="examples/ntop/", env=global_env)


# --- zh/examples/heart.md ---
@pytest.mark.skip(reason="Commented out in original script")
def test_heart(global_env):
    commands = [
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/heart/heart_dataset.tar",
        "tar -xvf heart_dataset.tar",
        "python forward.py 2>&1 | tee heart.log",
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/heart/heart_dataset.tar",
        "tar -xvf heart_dataset.tar",
        "python inverse.py TRAIN.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/heart/inverse_pretrained.pdparams",
    ]
    run_cmds(commands, cwd="examples/heart/", env=global_env)


# --- zh/examples/heat_exchanger.md ---
def test_heat_exchanger(global_env):
    commands = [
        "python heat_exchanger.py 2>&1 | tee heat_exchanger.log",
    ]
    run_cmds(commands, cwd="examples/heat_exchanger/", env=global_env)


# --- zh/examples/heat_pinn.md ---
def test_heat_pinn(global_env):
    commands = [
        "python heat_pinn.py 2>&1 | tee heat_pinn.log",
    ]
    run_cmds(commands, cwd="examples/heat_pinn/", env=global_env)


# --- zh/examples/phygeonet.md ---
def test_phygeonet(global_env):
    commands = [
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyGeoNet/heat_equation.npz -P ./data/",
        "python heat_equation.py 2>&1 | tee phygeonet.log",
    ]
    run_cmds(commands, cwd="examples/phygeonet/", env=global_env)


# --- zh/examples/chip_heat.md ---
def test_chip_heat(global_env):
    commands = [
        "python chip_heat.py 2>&1 | tee chip_heat.log",
    ]
    run_cmds(commands, cwd="examples/chip_heat/", env=global_env)


# --- zh/examples/hpinns.md ---
def test_hpinns(global_env):
    commands = [
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/hPINNs/hpinns_holo_train.mat -P ./datasets/",
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/hPINNs/hpinns_holo_valid.mat -P ./datasets/",
        "python holography.py 2>&1 | tee hpinns.log",
    ]
    run_cmds(commands, cwd="examples/hpinns/", env=global_env)


# --- zh/examples/cgcnn.md ---
@pytest.mark.skip(reason="Commented out in original script")
def test_cgcnn(global_env):
    commands = [
        "python CGCNN.py 2>&1 | tee cgcnn.log",
    ]
    run_cmds(commands, cwd="examples/cgcnn/", env=global_env)


# --- zh/examples/perovskite_solar_cells_nn.md ---
@pytest.mark.skip(reason="Commented out in original script")
def test_perovskite(global_env):
    commands = [
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/psc/data.zip",
        "unzip -o paddlescience/datasets/psc/data.zip",
        "python psc_nn.py mode=train 2>&1 | tee perovskite_solar_cells_nn.log",
    ]
    run_cmds(commands, cwd="examples/perovskite_solar_cells_nn/", env=global_env)


# --- zh/examples/MLP_LI.md ---
@pytest.mark.skip(reason="Commented out in original script")
def test_mlp_li(global_env):
    commands = [
        "python allen_cahn_piratenet.py 2>&1 | tee MLP_LI.log",
    ]
    run_cmds(commands, cwd="examples/MLP_LI/", env=global_env)


# --- en/examples/ml2ddb.md ---
@pytest.mark.skip(reason="Commented out in original script")
def test_ml2ddb(global_env):
    # Original script reuses allen_cahn logic here
    commands = [
        "python allen_cahn_piratenet.py 2>&1 | tee allen_cahn.log",
    ]
    run_cmds(commands, cwd="examples/allen_cahn/", env=global_env)


# --- zh/examples/extformer_moe.md ---
@pytest.mark.skip(reason="Commented out in original script")
def test_extformer_moe(global_env):
    commands = [
        "python allen_cahn_piratenet.py 2>&1 | tee allen_cahn.log",
    ]
    run_cmds(commands, cwd="examples/allen_cahn/", env=global_env)


# --- zh/examples/fourcastnet.md ---
@pytest.mark.skip(reason="Commented out in original script")
def test_fourcastnet(global_env):
    commands = [
        "python allen_cahn_piratenet.py 2>&1 | tee allen_cahn.log",
    ]
    run_cmds(commands, cwd="examples/allen_cahn/", env=global_env)


# --- zh/examples/nowcastnet.md ---
@pytest.mark.skip(reason="Commented out in original script")
def test_nowcastnet(global_env):
    commands = [
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/nowcastnet/mrms.tar",
        "mkdir -p ./datasets",
        "tar -xvf mrms.tar -C ./datasets/",
        "python nowcastnet.py mode=infer 2>&1 | tee nowcastne.log",
    ]
    run_cmds(commands, cwd="examples/nowcastnet", env=global_env)


# --- zh/examples/dgmr.md ---
@pytest.mark.skip(reason="Commented out in original script")
def test_dgmr(global_env):
    # This involves git lfs which might be tricky in CI, keeping as is
    commands = [
        "mkdir -p openclimatefix/nimrod-uk-1km/20200718/valid/subsampled_tiles_256_20min_stride",
        # Note: changing directory within a command sequence in run_cmds isn't supported like shell.
        # So we construct full paths or use separate run_cmds calls.
        # Here we simplify assuming the git lfs commands are run inside the target folder
        "cd openclimatefix/nimrod-uk-1km/20200718/valid/subsampled_tiles_256_20min_stride && git lfs install && git lfs pull --include='seq-24-*-of-00033.tfrecord.gz'",
        "python dgmr.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/dgmr/dgmr_pretrained.pdparams 2>&1 | tee dgm.log",
    ]
    run_cmds(commands, cwd="examples/dgmr", env=global_env)


# --- zh/examples/stafnet.md ---
@pytest.mark.skip(reason="Commented out in original script")
def test_stafnet(global_env):
    commands = [
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/stafnet/val_data.pkl -P ./dataset/",
        "python stafnet.py mode=eval EVAL.pretrained_model_path='https://paddle-org.bj.bcebos.com/paddlescience/models/stafnet/stafnet.pdparams' 2>&1 | tee stafne.log",
    ]
    run_cmds(commands, cwd="examples/stafnet", env=global_env)


# --- zh/examples/earthformer.md ---
@pytest.mark.skip(reason="Commented out in original script")
def test_earthformer(global_env):
    commands = [
        "python allen_cahn_piratenet.py 2>&1 | tee allen_cahn.log",
    ]
    run_cmds(commands, cwd="examples/allen_cahn/", env=global_env)


# --- zh/examples/graphcast.md ---
@pytest.mark.skip(reason="Commented out in original script")
def test_graphcast(global_env):
    commands = [
        "python -m pip install trimesh xarray rtree -i https://pypi.tuna.tsinghua.edu.cn/simple",
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/graphcast/dataset.zip",
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/graphcast/dataset-step12.zip",
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/models/graphcast/params.zip",
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/models/graphcast/template_graph.zip",
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/graphcast/stats.zip",
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/graphcast/graphcast-jax2paddle.csv -P ./data/",
        "unzip -o -q dataset.zip -d data/",
        "unzip -o -q dataset-step12.zip -d data/",
        "unzip -o -q params.zip -d data/",
        "unzip -o -q stats.zip -d data/",
        "unzip -o -q template_graph.zip -d data/",
        "python graphcast.py mode=eval EVAL.pretrained_model_path='data/params/GraphCast_small---ERA5-1979-2015---resolution-1.0---pressure-levels-13---mesh-2to5---precipitation-input-and-output.pdparams' 2>&1 | tee graphcas.log",
    ]
    run_cmds(commands, cwd="examples/graphcast", env=global_env)


# --- zh/examples/gencast.md ---
@pytest.mark.skip(reason="Commented out in original script")
def test_gencast(global_env):
    commands = [
        "python allen_cahn_piratenet.py 2>&1 | tee allen_cahn.log",
    ]
    run_cmds(commands, cwd="examples/gencast/", env=global_env)


# --- zh/examples/velocity_gan.md ---
@pytest.mark.skip(reason="Commented out in original script")
def test_velocity_gan(global_env):
    commands = [
        "python allen_cahn_piratenet.py 2>&1 | tee velocityGAN.log",
    ]
    run_cmds(commands, cwd="examples/velocityGAN/", env=global_env)


# --- zh/examples/tgcn.md ---
@pytest.mark.skip(reason="Commented out in original script")
def test_tgcn(global_env):
    commands = [
        "wget -cn https://paddle-org.bj.bcebos.com/paddlescience/datasets/tgcn/tgcn_data.zip",
        "unzip -o tgcn_data.zip",
        "wget -cn https://paddle-org.bj.bcebos.com/paddlescience/models/tgcn/PEMSD8_pretrained_model.pdparams",
        "python run.py data_name=PEMSD8 mode=eval EVAL.pretrained_model_path=PEMSD8_pretrained_model.pdparams 2>&1 | tee tgcn.log",
    ]
    run_cmds(commands, cwd="examples/tgcn/", env=global_env)


# --- zh/examples/iops.md ---
@pytest.mark.skip(reason="Commented out in original script")
def test_iops(global_env):
    commands = [
        "python allen_cahn_piratenet.py 2>&1 | tee allen_cahn.log",
    ]
    run_cmds(commands, cwd="examples/allen_cahn/", env=global_env)


# --- zh/examples/pangu_weather.md ---
@pytest.mark.skip(reason="Commented out in original script")
def test_pangu_weather(global_env):
    commands = [
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/models/Pangu/input_surface.npy -P ./data",
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/models/Pangu/input_upper.npy -P ./data",
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/models/Pangu/pangu_weather_1.onnx -P ./inference",
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/models/Pangu/pangu_weather_3.onnx -P ./inference",
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/models/Pangu/pangu_weather_6.onnx -P ./inference",
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/models/Pangu/pangu_weather_24.onnx -P ./inference",
        "python predict.py INFER.export_path=inference/pangu_weather_1 2>&1 | tee pangu_weather.log",
        "python predict.py INFER.export_path=inference/pangu_weather_3",
        "python predict.py INFER.export_path=inference/pangu_weather_6",
        "python predict.py INFER.export_path=inference/pangu_weather_24",
    ]
    run_cmds(commands, cwd="examples/pangu_weather/", env=global_env)


# --- zh/examples/fengwu.md ---
@pytest.mark.skip(reason="Commented out in original script")
def test_fengwu(global_env):
    commands = [
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/models/Fengwu/input1.npy -P ./data",
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/models/Fengwu/input2.npy -P ./data",
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/models/Fengwu/fengwu_v2.onnx -P ./inference 2>&1 | tee fengwu.log",
        "python predict.py",
    ]
    run_cmds(commands, cwd="examples/fengwu/", env=global_env)


# --- zh/examples/fuxi.md ---
@pytest.mark.skip(reason="Commented out in original script")
def test_fuxi(global_env):
    commands = [
        "unzip -o Sample_Data.zip 2>&1 | tee fuxi.log",
        "unzip -o FuXi_EC.zip",
        # Note: Script mentions modifying paths in config, which is manual.
        # This test assumes requirements are met or config is already correct.
        "python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple",
        "python predict.py",
    ]
    run_cmds(commands, cwd="examples/fuxi/", env=global_env)


# --- zh/examples/unetformer.md ---
@pytest.mark.skip(reason="Commented out in original script")
def test_unetformer(global_env):
    commands = [
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/unetformer/test.zip -P ./data/vaihingen/",
        "unzip -o -q ./data/vaihingen/test.zip -d data/vaihingen/",
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/models/unetformer/unetformer-r18-512-crop-ms-e105_epoch0_best.pdparams -P ./model_weights/vaihingen/unetformer-r18-512-crop-ms-e105/",
        "python vaihingen_test.py -c config/vaihingen/unetformer.py -o fig_results/vaihingen/unetformer --rgb 2>&1 | tee unetformer.log",
    ]
    run_cmds(commands, cwd="examples/unetformer/", env=global_env)


# --- zh/examples/wgan_gp.md ---
@pytest.mark.skip(reason="Commented out in original script")
def test_wgan_gp(global_env):
    # Fixed typo in original script comment "allen_cahn_piratenet.p2>&1|1"
    commands = [
        "python allen_cahn_piratenet.py 2>&1 | tee wgan.log",
    ]
    run_cmds(commands, cwd="examples/wgangp/", env=global_env)


# --- zh/examples/UTAE.md ---
@pytest.mark.skip(reason="Commented out in original script")
def test_utae(global_env):
    commands = [
        "python -m pip install geopandas -i https://pypi.tuna.tsinghua.edu.cn/simple",
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/models/utae/semantic.pdparams -P ./pretrained/",
        # Fixed formatting of the long command
        "python test_semantic.py --weight_file ./pretrained/semantic.pdparams --dataset_folder './data/PASTIS' --device gpu --num_workers 0 2>&1 | tee UTAE.log",
    ]
    run_cmds(commands, cwd="examples/UTAE/", env=global_env)


# --- zh/examples/smc_reac.md ---
@pytest.mark.skip(reason="Commented out in original script")
def test_smc_reac(global_env):
    commands = [
        "python -m pip install rdkit -i https://pypi.tuna.tsinghua.edu.cn/simple",
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/SMCReac/data_set.xlsx",
        "python smc_reac.py 2>&1 | tee allen_cahn.log",
    ]
    run_cmds(commands, cwd="examples/allen_cahn/", env=global_env)


# --- zh/examples/moflow.md ---
@pytest.mark.skip(reason="Commented out in original script")
def test_moflow(global_env):
    commands = [
        "python allen_cahn_piratenet.py 2>&1 | tee allen_cahn.log",
    ]
    run_cmds(commands, cwd="examples/allen_cahn/", env=global_env)


# --- zh/examples/ifm.md ---
@pytest.mark.skip(reason="Commented out in original script")
def test_ifm(global_env):
    commands = [
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/IFM/dataset.zip",
        "unzip -o dataset.zip",
        "python ifm.py mode=train data_label=tox21 MODEL.embed_name='IFM' 2>&1 | tee ifm.log",
    ]
    run_cmds(commands, cwd="examples/ifm/", env=global_env)


# --- zh/examples/synthemol.md ---
@pytest.mark.skip(reason="Commented out in original script")
def test_synthemol(global_env):
    commands = [
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/synthemol/Data.zip -P ./data/",
        "unzip -o ./data/Data.zip -d ./data/",
        "python main.py mode=train 2>&1 | tee synthemol.log",
    ]
    run_cmds(commands, cwd="examples/synthemol/", env=global_env)


# --- zh/examples/tadf.md ---
def test_tadf(global_env):
    # Note: original script did 'pushd examples/tadf/ && cd TADF_Est'.
    # We run commands in 'examples/tadf/TADF_Est' directly.
    commands = [
        "wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/TADF/Est/Est.dat https://paddle-org.bj.bcebos.com/paddlescience/datasets/TADF/smis.txt",
        "python -m pip install -r ../requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple",
        "python Est.py mode=train 2>&1 | tee tadf.log",
    ]
    run_cmds(commands, cwd="examples/tadf/TADF_Est", env=global_env)
