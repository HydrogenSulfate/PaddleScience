from __future__ import annotations

import os

import paddle
import pyvista as pv
from paddle_geometric.data import Data

# from paddle_geometric.io.fs import paddle_save
from tqdm import tqdm


def stack_pressure_data(
    base_path, order_file, max_num=None, pressure_key="p", output_dir="output"
):
    """
    Process VTK files in the order specified by order_file, stack coordinates and pressure,
    and save to a NetCDF file.

    Parameters:
    -----------
    base_path : str
        Root directory containing VTK folders (e.g., '/cluster/work/math/camlab-data/drivaernet/PressureVTK').
    order_file : str
        Path to order.txt specifying the processing sequence.
    max_num : int, optional
        only processing top max_num lines in the order_file (default: "200")
    pressure_key : str, optional
        Key for pressure in VTK point_data (default: pressure 'p').
    output_dir : str, optional
        Directory for the output file (default: 'output').

    Returns:
    --------
    None
    """

    folder_mapping = {
        "E_S_WWC_WM": ["E_S_WWC_WM"],
        "E_S_WW_WM": ["E_S_WW_WM"],
        "F_D_WM_WW": [
            "F_D_WM_WW_1",
            "F_D_WM_WW_2",
            "F_D_WM_WW_3",
            "F_D_WM_WW_4",
            "F_D_WM_WW_5",
            "F_D_WM_WW_6",
            "F_D_WM_WW_7",
            "F_D_WM_WW_8",
        ],
        "F_S_WWC_WM": ["F_S_WWC_WM"],
        "F_S_WWS_WM": ["F_S_WWS_WM"],
        "N_S_WWC_WM": ["N_S_WWC_WM"],
        "N_S_WWS_WM": ["N_S_WWS_WM"],
        "N_S_WW_WM": ["N_S_WW_WM"],
    }

    with open(order_file, "r") as f:
        order_list = f.read().splitlines()
    if max_num is not None:
        order_list = order_list[:max_num]

    for line in tqdm(order_list, desc="Processing VTK files"):
        # Parse folder name and number (e.g., 'N_S_WW_WM_633' -> 'N_S_WW_WM', '633')
        parts = line.rsplit("_", 1)
        if len(parts) != 2:
            print(f"Invalid format in order.txt: {line}")
            continue
        folder_name, number = parts

        if folder_name not in folder_mapping:
            print(f"unknown model: {folder_name} in order.txt")
            continue
        possible_folders = folder_mapping[folder_name]

        file_path = None
        for folder in possible_folders:
            vtk_filename = f"{folder_name}_{number}.vtk"
            vtk_filepath = os.path.join(base_path, folder, vtk_filename)

            if os.path.exists(vtk_filepath):
                file_path = vtk_filepath
                break
        if file_path is None:
            print(f"File not found {folder_name}_{number}: {possible_folders}")
            continue

        mesh = pv.read(file_path)

        # Compute point normals
        mesh = mesh.compute_normals(
            cell_normals=False, point_normals=True, auto_orient_normals=True
        )

        coords = mesh.points  # (sample_size, 3)
        normals = mesh.point_data["Normals"]  # (sample_size, 3)
        x = mesh.point_data[pressure_key]  # (sample_size,)
        data = Data(
            pos=paddle.to_tensor(
                coords, dtype=paddle.float32, place="cpu"
            ),  # (sample_size, 3)
            x=paddle.to_tensor(x, dtype=paddle.float32, place="cpu").unsqueeze(
                -1
            ),  # (sample_size, 1)
            c=paddle.to_tensor(
                normals, dtype=paddle.float32, place="cpu"
            ),  # (sample_size, 3) - normal vectors
        )
        data.filename = f"{folder_name}_{number}"
        processed_dir = output_dir
        os.makedirs(processed_dir, exist_ok=True)
        save_path = os.path.join(processed_dir, f"{folder_name}_{number}.pd")
        # paddle.save(data, save_path)
        paddle.save(
            {
                "pos": data.pos,
                "x": data.x,
                "c": data.c,
            },
            save_path,
        )


if __name__ == "__main__":
    base_path = "dataset/drivaernet/PressureVTK"
    order_file = "dataset/drivaernet/train.txt"
    pressure_key = "p"
    max_num = None
    output_dir = "dataset/drivaernet/processed_pyg_normal/train"

    stack_pressure_data(
        base_path=base_path,
        order_file=order_file,
        max_num=max_num,
        pressure_key=pressure_key,
        output_dir=output_dir,
    )
    # with paddle.device("cpu"):
    #     data = paddle.load("/work/EquationFlow/PaddleScience_GIFM/examples/_gaot3d/dataset/drivaernet/processed_pyg_normal/E_S_WW_WM_700.pd")
    #     for k, v in data.items():
    #         print(k, v.shape, v.dtype, v.place)

    # from ppsci.visualize import save_vtp_from_dict

    # mesh = pv.read('/work/EquationFlow/PaddleScience_GIFM/examples/_gaot3d/DrivAerPlusPlus/DrivAerPlusPlus/DrivAerNet++ Pressure/PressureVTK/N_S_WWS_WM/N_S_WWS_WM_001.vtk')
    # mesh = mesh.compute_normals(cell_normals=False, point_normals=True, auto_orient_normals=True)

    # coords = mesh.points  # (sample_size, 3)
    # normals = mesh.point_data["Normals"]  # (sample_size, 3)
    # x = mesh.point_data["p"]  # (sample_size,)
    # data = Data(
    #     pos=paddle.tensor(coords, dtype=paddle.float32),  # (sample_size, 3)
    #     x=paddle.tensor(x, dtype=paddle.float32).unsqueeze(-1),  # (sample_size, 1)
    #     c=paddle.tensor(
    #         normals, dtype=paddle.float32
    #     ),  # (sample_size, 3) - normal vectors
    # )
    # save_vtp_from_dict(
    #     "./N_S_WWS_WM_001",
    #     {
    #         "x": coords[:, 0:1],
    #         "y": coords[:, 1:2],
    #         "z": coords[:, 2:3],
    #         "n_x": normals[:, 0:1],
    #         "n_y": normals[:, 0:1],
    #         "n_z": normals[:, 0:1],
    #         "p": x,
    #     },
    #     ("x", "y", "z"),
    #     ("x", "y", "z", "n_x", "n_y", "n_z", "p"),
    # )
