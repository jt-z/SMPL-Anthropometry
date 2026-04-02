"""
从深度相机的 3D 关键点拟合 SMPL 模型，然后用 SMPL-Anthropometry 测量体型。

用法:
    python fit_smpl_from_keypoints.py \
        --npz /path/to/smpl_input.npz \
        --gender NEUTRAL \
        --output_dir ./fit_output

流程:
    1. 加载 3D 关键点 (COCO-17 格式, mm)
    2. 优化 SMPL beta + pose 参数以匹配 3D 关键点
    3. 用 SMPL-Anthropometry 库测量体型
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import smplx
from pprint import pprint

# 确保能 import 本库
sys.path.insert(0, os.path.dirname(__file__))
from measure import MeasureBody


# ─────────────────────────────────────────────
# COCO-17  →  SMPL-24 关节对应关系
#   COCO index : SMPL joint name
# ─────────────────────────────────────────────
COCO2SMPL = {
    5:  'left_shoulder',   # SMPL 16
    6:  'right_shoulder',  # SMPL 17
    7:  'left_elbow',      # SMPL 18
    8:  'right_elbow',     # SMPL 19
    9:  'left_wrist',      # SMPL 20
    10: 'right_wrist',     # SMPL 21
    11: 'left_hip',        # SMPL  1
    12: 'right_hip',       # SMPL  2
    13: 'left_knee',       # SMPL  4
    14: 'right_knee',      # SMPL  5
    15: 'left_ankle',      # SMPL  7
    16: 'right_ankle',     # SMPL  8
}

SMPL_JOINT2IND = {
    'pelvis': 0, 'left_hip': 1, 'right_hip': 2, 'spine1': 3,
    'left_knee': 4, 'right_knee': 5, 'spine2': 6,
    'left_ankle': 7, 'right_ankle': 8, 'spine3': 9,
    'left_foot': 10, 'right_foot': 11, 'neck': 12,
    'left_collar': 13, 'right_collar': 14, 'head': 15,
    'left_shoulder': 16, 'right_shoulder': 17,
    'left_elbow': 18, 'right_elbow': 19,
    'left_wrist': 20, 'right_wrist': 21,
    'left_hand': 22, 'right_hand': 23,
}


def load_keypoints(npz_path):
    """加载关键点，返回 (17,3) 米制坐标 和 (17,) 有效掩码"""
    d = np.load(npz_path)
    kps_mm = d['keypoints_3d'][:, :3]   # (17,3)  mm
    valid  = d['keypoints_valid']        # (17,)   bool
    kps_m  = kps_mm / 1000.0            # → 米
    return kps_m, valid


def build_coco_to_smpl_pairs(valid):
    """
    返回两个列表:
        coco_indices  : 观测关键点在 COCO-17 中的索引
        smpl_indices  : 对应 SMPL 关节索引
    只保留 valid==True 的对应关系
    """
    coco_indices, smpl_indices = [], []
    for coco_idx, joint_name in COCO2SMPL.items():
        if valid[coco_idx]:
            coco_indices.append(coco_idx)
            smpl_indices.append(SMPL_JOINT2IND[joint_name])
    return coco_indices, smpl_indices


def fit_smpl(kps_m, valid, gender, model_root, num_betas=10,
             lr=0.01, n_iter=500, verbose=True):
    """
    用梯度下降拟合 SMPL 的 betas 和 body_pose。

    Args:
        kps_m   : (17,3) float64, 米制 3D 关键点
        valid   : (17,)  bool
        gender  : 'MALE' | 'FEMALE' | 'NEUTRAL'
        model_root: str, data/ 目录

    Returns:
        betas : torch.tensor (1,10)
        model_output : smplx model output
    """
    device = torch.device('cpu')

    coco_idx, smpl_idx = build_coco_to_smpl_pairs(valid)
    if len(coco_idx) < 4:
        raise RuntimeError("有效关键点不足（< 4），无法拟合")

    obs = torch.tensor(kps_m[coco_idx], dtype=torch.float32, device=device)  # (K,3)
    smpl_idx_t = torch.tensor(smpl_idx, dtype=torch.long, device=device)

    # 去均值（只用观测到的髋关节估计中心）
    hip_coco = [c for c in [11, 12] if valid[c]]
    center_obs = torch.tensor(
        kps_m[hip_coco].mean(axis=0) if hip_coco else kps_m[coco_idx].mean(axis=0),
        dtype=torch.float32)
    obs_centered = obs - center_obs

    # 创建 SMPL 模型
    model = smplx.create(
        model_path=model_root,
        model_type='smpl',
        gender=gender,
        use_face_contour=False,
        num_betas=num_betas,
        ext='pkl'
    ).to(device)

    # 可优化参数
    betas      = nn.Parameter(torch.zeros(1, num_betas, device=device))
    body_pose  = nn.Parameter(torch.zeros(1, 69, device=device))   # 23 joints × 3
    global_ori = nn.Parameter(torch.zeros(1, 3, device=device))

    optimizer = torch.optim.Adam([betas, body_pose, global_ori], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.3)

    best_loss = float('inf')
    best_betas = betas.detach().clone()

    for i in range(n_iter):
        optimizer.zero_grad()

        out = model(betas=betas,
                    body_pose=body_pose,
                    global_orient=global_ori,
                    return_verts=True)

        joints = out.joints.squeeze()           # (24+, 3)
        joints_smpl = joints[:24]               # 取前 24 个 SMPL 关节
        pred = joints_smpl[smpl_idx_t]          # (K,3)

        # 去均值对齐
        hip_smpl_idx = [SMPL_JOINT2IND['left_hip'], SMPL_JOINT2IND['right_hip']]
        center_pred = joints_smpl[hip_smpl_idx].mean(dim=0)
        pred_centered = pred - center_pred

        # 关节位置损失 + beta 正则化
        joint_loss = ((pred_centered - obs_centered) ** 2).mean()
        reg_loss   = (betas ** 2).mean() * 0.01
        loss       = joint_loss + reg_loss

        loss.backward()
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_betas = betas.detach().clone()

        if verbose and (i % 100 == 0 or i == n_iter - 1):
            print(f"  iter {i:4d}  joint_loss={joint_loss.item()*1000:.2f}mm  "
                  f"total={loss.item()*1000:.2f}mm")

    print(f"\n最优关节误差: {best_loss*1000:.1f} mm")

    # 用最优 betas 重新推理（pose 用零姿势，只保留体型）
    with torch.no_grad():
        final_out = model(betas=best_betas,
                         body_pose=torch.zeros(1, 69),
                         global_orient=torch.zeros(1, 3),
                         return_verts=True)

    return best_betas, final_out


def measure_from_betas(betas, gender, model_root, measurement_names=None):
    """调用 SMPL-Anthropometry 进行体型测量"""
    measurer = MeasureBody('smpl')

    # MeasureSMPL 默认从 data/ 目录找模型
    measurer.from_body_model(gender=gender, shape=betas)

    if measurement_names is None:
        measurement_names = measurer.all_possible_measurements

    measurer.measure(measurement_names)
    # label_measurements 需要 {label: name} dict，用名字本身作 label
    measurer.label_measurements({n: n for n in measurement_names})

    return measurer


def main():
    parser = argparse.ArgumentParser(description='从 3D 关键点拟合 SMPL 并测量体型')
    parser.add_argument('--npz',
        default='/home/zjt/dev/On_Git_Projects/3D-Human-Measure/demo3_back_test/'
                'output_v2_yolov8-seg(segmenters_yolov8n-seg.pt)/smpl_input.npz',
        help='smpl_input.npz 路径')
    parser.add_argument('--gender', default='NEUTRAL',
        choices=['NEUTRAL', 'MALE', 'FEMALE'],
        help='性别 (需与模型文件匹配)')
    parser.add_argument('--model_root', default='data',
        help='SMPL-Anthropometry data/ 目录')
    parser.add_argument('--n_iter', type=int, default=500,
        help='优化迭代次数')
    parser.add_argument('--output_dir', default='./fit_output',
        help='结果输出目录')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. 加载数据 ──
    print(f"加载关键点: {args.npz}")
    kps_m, valid = load_keypoints(args.npz)
    print(f"  有效关键点: {valid.sum()}/17")

    # ── 2. 拟合 SMPL ──
    print(f"\n开始拟合 SMPL ({args.gender}), 迭代 {args.n_iter} 次...")
    betas, model_output = fit_smpl(
        kps_m, valid, args.gender, args.model_root, n_iter=args.n_iter)

    # 保存 betas
    betas_np = betas.detach().numpy()
    betas_path = os.path.join(args.output_dir, 'betas.npy')
    np.save(betas_path, betas_np)
    print(f"\nbetas 已保存: {betas_path}")
    print(f"betas: {betas_np.squeeze().round(3)}")

    # ── 3. 体型测量 ──
    print("\n开始体型测量...")
    try:
        measurer = measure_from_betas(betas, args.gender, args.model_root)
        measurer.label_measurements(measurer.all_possible_measurements)

        print("\n测量结果 (cm):")
        print("-" * 40)
        for name, val in sorted(measurer.measurements.items()):
            print(f"  {name:<35s}: {val:.1f} cm")

        # 保存结果
        result_path = os.path.join(args.output_dir, 'measurements.txt')
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write(f"性别: {args.gender}\n")
            f.write(f"betas: {betas_np.squeeze().round(3)}\n\n")
            f.write("测量结果 (cm):\n")
            for name, val in sorted(measurer.measurements.items()):
                f.write(f"  {name}: {val:.1f}\n")
        print(f"\n结果已保存: {result_path}")

    except Exception as e:
        print(f"\n测量失败: {e}")
        print("请确认 data/smpl/ 目录下有 SMPL 模型 .pkl 文件")
        print("下载地址: https://smpl.is.tue.mpg.de/")


if __name__ == '__main__':
    main()
