"""
在浏览器中 3D 查看 SMPL 人体模型及体型测量结果。

用法:
    # 使用已有 betas（最快）
    python view_smpl_3d.py --betas ./fit_output/betas.npy

    # 使用 smpl_params.npz（fit_smpl_from_data.py 输出）
    python view_smpl_3d.py --params ./output_smpl_fit/smpl_params.npz

    # 只看体型，不显示测量线
    python view_smpl_3d.py --betas ./fit_output/betas.npy --no_measurements

    # 同时保存 HTML 文件（可离线打开）
    python view_smpl_3d.py --betas ./fit_output/betas.npy --save_html ./fit_output/body_3d.html
"""

import argparse
import os
import numpy as np
import torch

import sys
sys.path.insert(0, os.path.dirname(__file__))
from measure import MeasureBody


def load_betas(betas_path=None, params_path=None):
    if betas_path:
        arr = np.load(betas_path)
        betas = torch.tensor(arr, dtype=torch.float32)
        if betas.dim() == 1:
            betas = betas.unsqueeze(0)
        return betas
    if params_path:
        d = np.load(params_path)
        arr = d['betas']
        betas = torch.tensor(arr, dtype=torch.float32)
        if betas.dim() == 1:
            betas = betas.unsqueeze(0)
        return betas
    raise ValueError("需要提供 --betas 或 --params 参数")


def main():
    parser = argparse.ArgumentParser(description="3D 浏览器查看 SMPL 人体模型")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--betas",  default="./fit_output/betas.npy",
                     help="betas.npy 路径（fit_smpl_from_keypoints.py 输出）")
    src.add_argument("--params", default=None,
                     help="smpl_params.npz 路径（fit_smpl_from_data.py 输出）")

    parser.add_argument("--gender", default="NEUTRAL",
                        choices=["NEUTRAL", "MALE", "FEMALE"])
    parser.add_argument("--model_type", default="smpl",
                        choices=["smpl", "smplx"])
    parser.add_argument("--no_measurements", action="store_true",
                        help="不绘制测量线（加载更快）")
    parser.add_argument("--no_landmarks",    action="store_true",
                        help="不显示关键点")
    parser.add_argument("--no_joints",       action="store_true",
                        help="不显示骨架关节")
    parser.add_argument("--save_html", default=None,
                        help="同时将 3D 模型保存为 HTML 文件，可离线打开")
    args = parser.parse_args()

    # ── 加载 betas ──
    betas = load_betas(betas_path=args.betas if not args.params else None,
                       params_path=args.params)
    print(f"betas: {betas.squeeze().numpy().round(3)}")

    # ── 重建 SMPL 模型 ──
    print(f"\n重建 {args.model_type.upper()} 模型 (gender={args.gender})...")
    measurer = MeasureBody(args.model_type)
    measurer.from_body_model(gender=args.gender, shape=betas)

    # ── 测量 ──
    measurement_names = measurer.all_possible_measurements
    measurer.measure(measurement_names)

    print(f"\n测量结果 ({len(measurer.measurements)} 项):")
    print("-" * 40)
    for name, val in sorted(measurer.measurements.items()):
        print(f"  {name:<35s}: {val:.1f} cm")

    # ── 3D 可视化（打开浏览器） ──
    print("\n打开浏览器 3D 查看...")
    title = (f"SMPL Body — Gender: {args.gender} | "
             f"Height: {measurer.measurements.get('height', 0):.1f} cm | "
             f"Chest: {measurer.measurements.get('chest circumference', 0):.1f} cm | "
             f"Waist: {measurer.measurements.get('waist circumference', 0):.1f} cm")

    measurer.visualize(
        measurement_names=[] if args.no_measurements else measurement_names,
        visualize_body=True,
        visualize_landmarks=not args.no_landmarks,
        visualize_joints=not args.no_joints,
        visualize_measurements=not args.no_measurements,
        title=title,
        save_html=args.save_html,
    )


if __name__ == "__main__":
    main()
