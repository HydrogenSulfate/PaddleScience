set -ex
export LD_PRELOAD=/usr/local/corex-4.3.8/lib64/libcuda.so.1
export PADDLE_XCCL_BACKEND=iluvatar_gpu
# export PYTHONPATH=$PYTHONPATH:/usr/local/corex-4.3.8/lib64/python3/dist-packages
# prepare environment
which python
export PYTHONPATH=`pwd`:$PYTHONPATH
# python -m pip install --upgrade pip
# python -m pip install uv
export MAX_ITERS=3
unset https_proxy http_proxy
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=1

# zh/examples/allen_cahn.md
# pushd examples/allen_cahn/
# python allen_cahn_piratenet.py 2>&1 | tee allen_cahn.log
# popd

# zh/examples/deephpms.md
pushd examples/deephpms/
wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/DeepHPMs/burgers_sine.mat -P ./datasets/
python burgers.py DATASET_PATH=./datasets/burgers_sine.mat DATASET_PATH_SOL=./datasets/burgers_sine.mat 2>&1 | tee deephpms.log
popd

# zh/examples/deeponet.md
pushd examples/operator_learning/
wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/DeepONet/antiderivative_unaligned_train.npz
wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/DeepONet/antiderivative_unaligned_test.npz
python deeponet.py 2>&1 | tee operator_learnin.log
popd

# zh/examples/euler_beam.md
pushd examples/euler_beam/
python euler_beam.py 2>&1 | tee euler_beam.log
popd

# zh/examples/laplace2d.md
pushd examples/laplace/
python laplace2d.py 2>&1 | tee laplace.log
popd

# zh/examples/lorenz.md
pushd examples/lorenz/
wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/lorenz_training_rk.hdf5 -P ./datasets/
wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/lorenz_valid_rk.hdf5 -P ./datasets/
python train_enn.py 2>&1 | tee lorenz.log
# python train_transformer.py
popd

# zh/examples/pirbn.md
pushd jointContribution/PIRBN
python main.py 2>&1 | tee PIRBN.log
popd

# zh/examples/rossler.md
pushd examples/rossler/
wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/rossler_training.hdf5 -P ./datasets/ 2>&1 | tee rossler.log
wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/rossler_valid.hdf5 -P ./datasets/
python train_enn.py
# python train_transformer.py
popd

# zh/examples/volterra_ide.md
pushd examples/ide/
python volterra_ide.py 2>&1 | tee ide.log
popd

# zh/examples/nlsmb.md
# pushd examples/NLS-MB/
# python NLS-MB_optical_rogue_wave.py 2>&1 | tee MB.log
# python NLS-MB_optical_soliton.py
# popd

# zh/examples/spinn.md
# pushd examples/spinn/
# python helmholtz3d.py 2>&1 | tee spinn.log
# popd

# zh/examples/xpinns.md
# pushd examples/xpinn/
# wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/XPINN/XPINN_2D_PoissonEqn.mat -P ./data/ 2>&1 | tee xpinn.log
# python xpinn.py
# popd

# zh/examples/neuraloperator.md
# pushd examples/neuraloperator/
# wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/neuraloperator/darcy_flow/darcy_train_16.npy -P ./datasets/darcyflow/
# wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/neuraloperator/darcy_flow/darcy_test_32.npy -P ./datasets/darcyflow/
# wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/neuraloperator/darcy_flow/darcy_test_16.npy -P ./datasets/darcyflow/
# python train_tfno.py 2>&1 | tee neuraloperator.log
# python train_uno.py
# popd

# zh/examples/brusselator3d.md
# pushd examples/brusselator3d/
# python brusselator3d.py 2>&1 | tee brusselator3d.log
# popd

# zh/examples/transformer4sr.md
# pushd examples/transformer4sr/
# pip install zss
# tar -xzvf data_generated.tar.gz
# python transformer4sr.py 2>&1 | tee transformer4s.log
# popd

# zh/examples/latent_no.md
# pushd examples/LatentNO/
# python LatentNO-steady.py --config-name=LatentNO-Darcy.yaml 2>&1 | tee LatentNO.log
# popd

# zh/examples/fundiff.md
# pushd examples/fundiff/
# python main.py -cn fae.yaml 2>&1 | tee fundiff.log
# popd

# zh/examples/catheter.md
# pushd examples/catheter/
# wget -c https://dataset.bj.bcebos.com/PaddleScience/2024%20AI-aided%20geometric%20design%20of%20anti-infection%20catheters/data.zip
# unzip -o data.zip
# python catheter.py 2>&1 | tee catheter.log
# popd

# zh/examples/amgnet.md
pushd examples/amgnet/
wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/AMGNet/data.zip
unzip -o data.zip
python amgnet_airfoil.py 2>&1 | tee amgnet.log
popd

# zh/examples/aneurysm.md
pushd examples/aneurysm/
wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/aneurysm/aneurysm_dataset.tar
tar -xvf aneurysm_dataset.tar
python aneurysm.py 2>&1 | tee aneurysm.log
popd

# zh/examples/bubble.md
pushd examples/bubble/
wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/BubbleNet/bubble.mat
python bubble.py 2>&1 | tee bubble.log
popd

# zh/examples/cfdgcn.md
# pushd examples/allen_cahn/
# python allen_cahn_piratenet.py 2>&1 | tee allen_cahn.log
# popd

# zh/examples/adv_cvit.md
# pushd examples/adv/
# wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/Cvit_adv/adv_a0.npy -P ./data
# wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/Cvit_adv/adv_aT.npy -P ./data
# python adv_cvit.py 2>&1 | tee adv.log
# popd

# zh/examples/ns_cvit.md
# pushd examples/allen_cahn/
# python allen_cahn_piratenet.py 2>&1 | tee allen_cahn.log
# popd

# zh/examples/cylinder2d_unsteady.md
pushd examples/cylinder/2d_unsteady/
wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/cylinder2d_unsteady_Re100/cylinder2d_unsteady_Re100_dataset.tar
tar -xvf cylinder2d_unsteady_Re100_dataset.tar
python cylinder2d_unsteady_Re100.py 2>&1 | tee 2d_unsteady.log
popd

# zh/examples/cylinder2d_unsteady_transformer_physx.md
pushd examples/cylinder/2d_unsteady/transformer_physx
wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/cylinder_training.hdf5 -P ./datasets/
wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/cylinder_valid.hdf5 -P ./datasets/
python train_enn.py 2>&1 | tee transformer_phys.log
# python train_transformer.py
popd

# zh/examples/darcy2d.md
pushd examples/darcy/
python darcy2d.py 2>&1 | tee darcy.log
popd

# zh/examples/deepcfd.md
pushd examples/deepcfd/
wget -c -P ./datasets/ https://paddle-org.bj.bcebos.com/paddlescience/datasets/DeepCFD/dataX.pkl
wget -c -P ./datasets/ https://paddle-org.bj.bcebos.com/paddlescience/datasets/DeepCFD/dataY.pkl
python deepcfd.py 2>&1 | tee deepcfd.log
popd

# zh/examples/drivaernet.md
# pushd examples/drivaernet/
# wget -c https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAer%2B%2B/data.tar
# tar -xvf data.tar
# python drivaernet.py 2>&1 | tee drivaernet.log
# popd

# zh/examples/drivaernetplusplus.md
# pushd examples/drivaernetplusplus/
# mkdir -p data/subset_dir
# wget -c https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAer%2B%2B/DrivAer%2B%2B_Points.tar
# tar -xvf DrivAer++_Points.tar -C ./data
# wget -c https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAer%2B%2B/DrivAerNetPlusPlus_Drag_8k.csv -P ./data
# wget -c https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAer%2B%2B/test_design_ids.txt -P ./data/subset_dir
# wget -c https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAer%2B%2B/train_design_ids.txt -P ./data/subset_dir
# wget -c https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAer%2B%2B/val_design_ids.txt -P ./data/subset_dir
# mv ./data/workspace/gino_data/14_DrivAer++/paddle_tensor ./data/DrivAerNetPlusPlus_Processed_Point_Clouds_100k_paddle
# rm -rf data/workspace
# python drivaernetplusplus.py 2>&1 | tee drivaernetplusplus.log
# popd

# zh/examples/ldc2d_steady.md
# pushd examples/ldc/
# wget -c -P ./data/
#     https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re100.mat \
#     https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re400.mat \
#     https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re1000.mat \
#     https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re3200.mat
# python ldc_2d_Re3200_sota.py 2>&1 | tee ldc.log
# popd

# zh/examples/ldc2d_unsteady.md
# pushd examples/ldc/
# python ldc2d_unsteady_Re10.py 2>&1 | tee ldc.log
# popd

# zh/examples/labelfree_DNN_surrogate.md
# pushd examples/aneurysm/
# wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/LabelFree-DNN-Surrogate/LabelFree-DNN-Surrogate_data.zip
# unzip -o LabelFree-DNN-Surrogate_data.zip
# python aneurysm_flow.py 2>&1 | tee aneurysm_flow.log
# popd

# zh/examples/nsfnet.md
pushd examples/nsfnet/
python VP_NSFNet1.py 2>&1 | tee nsfnet.log
popd

# zh/examples/phycrnet.md
# pushd examples/phycrnet/
# wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyCRNet/burgers_1501x2x128x128.mat -P ./data/
# python main.py DATA_PATH=./data/burgers_1501x2x128x128.mat 2>&1 | tee phycrnet.log
# popd

# zh/examples/shock_wave.md
# pushd examples/shock_wave/
# python shock_wave.py 2>&1 | tee shock_wave.log
# popd

# zh/examples/tempoGAN.md
# pushd examples/tempoGAN/
# wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/tempoGAN/2d_train.mat -P datasets/tempoGAN/
# wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/tempoGAN/2d_valid.mat -P datasets/tempoGAN/
# python tempoGAN.py 2>&1 | tee tempoGAN.log
# popd

# zh/examples/nsfnet4.md
# pushd examples/nsfnet/
# wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/NSFNet/NSF4_data.zip -P ./data/
# unzip -o ./data/NSF4_data.zip
# python VP_NSFNet4.py mode=eval data_dir=./data/ EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/nsfnet/nsfnet4.pdparams 2>&1 | tee nsfnet.log
# popd

# zh/examples/viv.md
pushd examples/fsi/
python viv.py 2>&1 | tee fsi.log
popd

# zh/examples/biharmonic2d.md
pushd examples/biharmonic2d/
python biharmonic2d.py 2>&1 | tee biharmonic2d.log
popd

# zh/examples/bracket.md
pushd examples/bracket/
wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/bracket/bracket_dataset.tar
tar -xvf bracket_dataset.tar
python bracket.py 2>&1 | tee bracket.log
popd

# zh/examples/control_arm.md
pushd examples/control_arm/
wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/control_arm/control_arm.stl -P ./datasets/
python forward_analysis.py 2>&1 | tee control_arm.log
popd

# zh/examples/epnn.md
pushd examples/epnn/
wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/epnn/dstate-16-plas.dat -P ./datasets/
wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/epnn/dstress-16-plas.dat -P ./datasets/
python epnn.py 2>&1 | tee epnn.log
popd

# zh/examples/phylstm.md
pushd examples/phylstm/
wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyLSTM/data_boucwen.mat
python phylstm2.py 2>&1 | tee phylstm.log
popd

# zh/examples/topopt.md
pushd examples/topopt/
wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/topopt/top_dataset.h5 -P ./datasets/
python topopt.py 2>&1 | tee topopt.log
popd

# zh/examples/ntopo.md
# pushd examples/ntop/
# python ntopo.py 2>&1 | tee ntop.log
# popd

# zh/examples/heart.md
# pushd examples/heart/
# wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/heart/heart_dataset.tar
# tar -xvf heart_dataset.tar
# python forward.py 2>&1 | tee heart.log

# wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/heart/heart_dataset.tar
# tar -xvf heart_dataset.tar
# python inverse.py TRAIN.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/heart/inverse_pretrained.pdparams
# popd

# zh/examples/heat_exchanger.md
pushd examples/heat_exchanger/
python heat_exchanger.py 2>&1 | tee heat_exchanger.log
popd

# zh/examples/heat_pinn.md
pushd examples/heat_pinn/
python heat_pinn.py 2>&1 | tee heat_pinn.log
popd

# zh/examples/phygeonet.md
pushd examples/phygeonet/
wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyGeoNet/heat_equation.npz -P ./data/
python heat_equation.py 2>&1 | tee phygeonet.log
popd

# zh/examples/chip_heat.md
pushd examples/chip_heat/
python chip_heat.py 2>&1 | tee chip_heat.log
popd

# zh/examples/hpinns.md
pushd examples/hpinns/
wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/hPINNs/hpinns_holo_train.mat -P ./datasets/
wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/hPINNs/hpinns_holo_valid.mat -P ./datasets/
python holography.py 2>&1 | tee hpinns.log
popd

# zh/examples/cgcnn.md
# pushd examples/cgcnn/
# python CGCNN.py 2>&1 | tee cgcnn.log
# popd

# zh/examples/perovskite_solar_cells_nn.md
# pushd examples/perovskite_solar_cells_nn/
# wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/psc/data.zip
# unzip -o paddlescience/datasets/psc/data.zip
# python psc_nn.py mode=train 2>&1 | tee perovskite_solar_cells_nn.log
# popd

# zh/examples/MLP_LI.md
# pushd examples/MLP_LI/
# python allen_cahn_piratenet.py 2>&1 | tee MLP_LI.log
# popd

# en/examples/ml2ddb.md
# pushd examples/allen_cahn/
# python allen_cahn_piratenet.py 2>&1 | tee allen_cahn.log
# popd

# zh/examples/extformer_moe.md
# pushd examples/allen_cahn/
# python allen_cahn_piratenet.py 2>&1 | tee allen_cahn.log
# popd

# zh/examples/fourcastnet.md
# pushd examples/allen_cahn/
# python allen_cahn_piratenet.py 2>&1 | tee allen_cahn.log
# popd

# zh/examples/nowcastnet.md
# pushd examples/nowcastnet
# wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/nowcastnet/mrms.tar
# mkdir ./datasets
# tar -xvf mrms.tar -C ./datasets/
# python nowcastnet.py mode=infer 2>&1 | tee nowcastne.log
# popd

# zh/examples/dgmr.md
# pushd examples/dgmr
# mkdir openclimatefix/nimrod-uk-1km/20200718/valid/subsampled_tiles_256_20min_stride
# cd openclimatefix/nimrod-uk-1km/20200718/valid/subsampled_tiles_256_20min_stride
# git lfs install
# git lfs pull --include="seq-24-*-of-00033.tfrecord.gz"
# python dgmr.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/dgmr/dgmr_pretrained.pdparams 2>&1 | tee dgm.log
# popd

# zh/examples/stafnet.md
# pushd examples/stafnet
# wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/stafnet/val_data.pkl -P ./dataset/
# python stafnet.py mode=eval EVAL.pretrained_model_path="https://paddle-org.bj.bcebos.com/paddlescience/models/stafnet/stafnet.pdparams" 2>&1 | tee stafne.log
# popd

# zh/examples/earthformer.md
# pushd examples/allen_cahn/
# python allen_cahn_piratenet.py 2>&1 | tee allen_cahn.log
# popd

# zh/examples/graphcast.md
# pushd examples/graphcast
# uv pip install --system trimesh xarray rtree
# wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/graphcast/dataset.zip
# wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/graphcast/dataset-step12.zip
# wget -c https://paddle-org.bj.bcebos.com/paddlescience/models/graphcast/params.zip
# wget -c https://paddle-org.bj.bcebos.com/paddlescience/models/graphcast/template_graph.zip
# wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/graphcast/stats.zip
# wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/graphcast/graphcast-jax2paddle.csv -P ./data/
# unzip -o -q dataset.zip -d data/
# unzip -o -q dataset-step12.zip -d data/
# unzip -o -q params.zip -d data/
# unzip -o -q stats.zip -d data/
# unzip -o -q template_graph.zip -d data/
# python graphcast.py mode=eval EVAL.pretrained_model_path="data/params/GraphCast_small---ERA5-1979-2015---resolution-1.0---pressure-levels-13---mesh-2to5---precipitation-input-and-output.pdparams" 2>&1 | tee graphcas.log
# popd

# zh/examples/gencast.md
# pushd examples/gencast/
# python allen_cahn_piratenet.py 2>&1 | tee allen_cahn.log
# popd

# zh/examples/velocity_gan.md
# pushd examples/velocityGAN/
# python allen_cahn_piratenet.py 2>&1 | tee velocityGAN.log
# popd

# zh/examples/tgcn.md
# pushd examples/tgcn/
# wget -cn https://paddle-org.bj.bcebos.com/paddlescience/datasets/tgcn/tgcn_data.zip
# unzip -o tgcn_data.zip
# wget -cn https://paddle-org.bj.bcebos.com/paddlescience/models/tgcn/PEMSD8_pretrained_model.pdparams
# python run.py data_name=PEMSD8 mode=eval EVAL.pretrained_model_path=PEMSD8_pretrained_model.pdparams 2>&1 | tee tgcn.log
# popd

# zh/examples/iops.md
# pushd examples/allen_cahn/
# python allen_cahn_piratenet.py 2>&1 | tee allen_cahn.log
# popd

# zh/examples/pangu_weather.md
# pushd examples/pangu_weather/
# wget -c https://paddle-org.bj.bcebos.com/paddlescience/models/Pangu/input_surface.npy -P ./data
# wget -c https://paddle-org.bj.bcebos.com/paddlescience/models/Pangu/input_upper.npy -P ./data

# # Download pretrain model weight
# wget -c https://paddle-org.bj.bcebos.com/paddlescience/models/Pangu/pangu_weather_1.onnx -P ./inference
# wget -c https://paddle-org.bj.bcebos.com/paddlescience/models/Pangu/pangu_weather_3.onnx -P ./inference
# wget -c https://paddle-org.bj.bcebos.com/paddlescience/models/Pangu/pangu_weather_6.onnx -P ./inference
# wget -c https://paddle-org.bj.bcebos.com/paddlescience/models/Pangu/pangu_weather_24.onnx -P ./inference

# # 1h interval-time model inference
# python predict.py INFER.export_path=inference/pangu_weather_1 2>&1 | tee pangu_weather.log
# # 3h interval-time model inference
# python predict.py INFER.export_path=inference/pangu_weather_3
# # 6h interval-time model inference
# python predict.py INFER.export_path=inference/pangu_weather_6
# # 24h interval-time model inference
# python predict.py INFER.export_path=inference/pangu_weather_24
# popd

# zh/examples/fengwu.md
# pushd examples/fengwu/
# wget -c https://paddle-org.bj.bcebos.com/paddlescience/models/Fengwu/input1.npy -P ./data
# wget -c https://paddle-org.bj.bcebos.com/paddlescience/models/Fengwu/input2.npy -P ./data

# # Download pretrain model weight
# wget -c https://paddle-org.bj.bcebos.com/paddlescience/models/Fengwu/fengwu_v2.onnx -P ./inference 2>&1 | tee fengwu.log
# inference
# python predict.py
# popd

# zh/examples/fuxi.md
# pushd examples/fuxi/
# unzip -o Sample_Data.zip 2>&1 | tee fuxi.log
# unzip -o FuXi_EC.zip
# modify the path of model and datasets in examples/fuxi/conf, and inference
# pip install -r requirements.txt
# python predict.py
# popd

# zh/examples/unetformer.md
# pushd examples/unetformer/
# wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/unetformer/test.zip -P ./data/vaihingen/
# unzip -o -q ./data/vaihingen/test.zip -d data/vaihingen/
# # 下载预训练模型文件
# wget -c https://paddle-org.bj.bcebos.com/paddlescience/models/unetformer/unetformer-r18-512-crop-ms-e105_epoch0_best.pdparams -P ./model_weights/vaihingen/unetformer-r18-512-crop-ms-e105/
# python vaihingen_test.py -c config/vaihingen/unetformer.py -o fig_results/vaihingen/unetformer --rgb 2>&1 | tee unetformer.log
# popd

# zh/examples/wgan_gp.md
# pushd examples/wgangp//
# python allen_cahn_piratenet.p2>&1|1 | tee /.log
# popd

# zh/examples/UTAE.md
# uv pip install geopandas --system
# pushd examples/UTAE/
# wget -c https://paddle-org.bj.bcebos.com/paddlescience/models/utae/semantic.pdparams -P ./pretrained/
# python test_semantic.py \
#     --weight_file ./pretrained/semantic.pdparams \
#     --dataset_folder "./data/PASTIS" \
#     --device gpu
#     --num_workers 0 2>&1 | tee UTAE.log
# popd

# zh/examples/smc_reac.md
# pushd examples/allen_cahn/
# uv pip install rdkit --system
# wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/SMCReac/data_set.xlsx
# python smc_reac.py 2>&1 | tee allen_cahn.log
# popd

# zh/examples/moflow.md
# pushd examples/allen_cahn/
# python allen_cahn_piratenet.py 2>&1 | tee allen_cahn.log
# popd

# zh/examples/ifm.md
# pushd examples/ifm/
# wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/IFM/dataset.zip
# unzip -o dataset.zip
# python ifm.py mode=train data_label=tox21 MODEL.embed_name='IFM' 2>&1 | tee ifm.log
# popd

# zh/examples/synthemol.md
# pushd examples/synthemol/
# wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/synthemol/Data.zip -P ./data/
# unzip -o ./data/Data.zip -d ./data/
# python main.py mode=train 2>&1 | tee synthemol.log
# popd

# zh/examples/tadf.md
pushd examples/tadf/
cd TADF_Est
wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/TADF/Est/Est.dat  https://paddle-org.bj.bcebos.com/paddlescience/datasets/TADF/smis.txt
python Est.py mode=train 2>&1 | tee tadf.log
popd
