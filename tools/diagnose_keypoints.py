#!/usr/bin/env python3
"""
诊断工具：分析YOLO关键点和SMPL关键点的差异
"""

import numpy as np
import torch
import smplx
import os
import re
from scipy.spatial.transform import Rotation as R

def load_smpl_model():
    """加载SMPL模型（零参数）"""
    model = smplx.create(
        model_path="data",
        model_type="smpl",
        gender="neutral",
        num_betas=10,
        use_face_contour=False,
        ext='pkl'
    )
    
    output = model(
        betas=torch.zeros(1, 10),
        body_pose=torch.zeros(1, 23*3),
        global_orient=torch.zeros(1, 3),
        transl=torch.zeros(1, 3),
        return_verts=True,
        return_joints=True
    )
    
    joints_smpl = output.joints.detach().numpy()[0]
    return joints_smpl

def parse_yolo_keypoints(txt_path):
    """解析YOLO的3D关键点"""
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    coco_keypoint_names = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    keypoints_dict = {}
    
    for kp_name in coco_keypoint_names:
        pattern = rf'{kp_name}\s+:.+?3D=\[\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\]'
        match = re.search(pattern, content)
        
        if match:
            x = float(match.group(1))
            y = float(match.group(2))
            z = float(match.group(3))
            keypoints_dict[kp_name] = np.array([x, y, z])
    
    return keypoints_dict

def print_statistics(name, points):
    """打印点云统计"""
    print(f"\n{name}:")
    print(f"  形状: {points.shape}")
    print(f"  X 范围: [{points[:,0].min():.2f}, {points[:,0].max():.2f}]")
    print(f"  Y 范围: [{points[:,1].min():.2f}, {points[:,1].max():.2f}]")
    print(f"  Z 范围: [{points[:,2].min():.2f}, {points[:,2].max():.2f}]")
    print(f"  尺度: {np.max(points, axis=0) - np.min(points, axis=0)}")

def main():
    txt_path = "/home/zjt/dev/On_Git_Projects/3D-Human-Measure/frame_output/frame_1860_yolo_measure_results.txt"
    
    print("="*60)
    print("YOLO vs SMPL 关键点差异诊断")
    print("="*60)
    
    # 1. 解析YOLO关键点
    yolo_kps = parse_yolo_keypoints(txt_path)
    print(f"\n解析到 {len(yolo_kps)} 个YOLO关键点:")
    for name, pt in yolo_kps.items():
        print(f"  {name:15s}: {pt}")
    
    # 转换为数组
    coco_names = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    yolo_array = np.array([yolo_kps[name] for name in coco_names if name in yolo_kps])
    
    # 2. 加载SMPL模型零参数关节
    smpl_joints = load_smpl_model()
    
    # SMPL到COCO的映射（仅躯干和四肢）
    smpl_to_coco = {
        0: 'nose',           # Pelvis → 骨盆，但没有
        1: 'left_hip',       # L_Hip
        2: 'right_hip',      # R_Hip
        3: 'spine_bottom',   # Spine3 (没在COCO里)
        4: 'left_knee',      # L_Knee
        5: 'right_knee',     # R_Knee
        6: 'spine_mid',      # Spine2 (没在COCO里)
        7: 'left_ankle',     # L_Ankle
        8: 'right_ankle',    # R_Ankle
        9: 'spine_top',      # Spine1 (没在COCO里)
        10: 'left_foot',     # L_Foot (没在COCO里)
        11: 'right_foot',    # R_Foot (没在COCO里)
        12: 'neck',           # Neck (没在COCO里)
        13: 'left_head_top', # L_Head (没在COCO里)
        14: 'right_head_top',# R_Head (没在COCO里)
        15: 'left_shoulder', # L_Shoulder
        16: 'right_shoulder',# R_Shoulder
        17: 'left_elbow',    # L_Elbow
        18: 'right_elbow',   # R_Elbow
        19: 'left_wrist',    # L_Wrist
        20: 'right_wrist',   # R_Wrist
        21: 'left_hand',     # L_Hand (没在COCO里)
        22: 'right_hand',    # R_Hand (没在COCO里)
    }
    
    # 提取SMPL中与COCO对应的关节
    common_names = [name for name in coco_names if name in smpl_to_coco.values()]
    smpl_common_indices = [idx for idx, name in smpl_to_coco.items() if name in coco_names]
    smpl_common = smpl_joints[smpl_common_indices]
    
    # 3. 打印统计信息
    print_statistics("YOLO 关键点 (单位: mm)", yolo_array)
    print_statistics("SMPL 关节 (零参数, 单位: m)", smpl_common)
    
    # 4. 计算缩放比例
    yolo_scale = np.max(yolo_array, axis=0) - np.min(yolo_array, axis=0)
    smpl_scale = np.max(smpl_common, axis=0) - np.min(smpl_common, axis=0)
    
    print(f"\n尺度对比:")
    print(f"  YOLO 高度: {yolo_scale[1]:.1f} mm")
    print(f"  SMPL 高度: {smpl_scale[1]*1000:.1f} mm")
    print(f"  缩放比例 (YOLO/SMPL): {(yolo_scale[1]/(smpl_scale[1]*1000)):.2f}")
    
    # 5. 计算肩宽对比
    if 'left_shoulder' in yolo_kps and 'right_shoulder' in yolo_kps:
        yolo_shoulder_width = np.linalg.norm(yolo_kps['left_shoulder'] - yolo_kps['right_shoulder'])
        
        # SMPL的肩宽
        smpl_ls = smpl_joints[15]
        smpl_rs = smpl_joints[16]
        smpl_shoulder_width = np.linalg.norm(smpl_ls - smpl_rs)
        
        print(f"\n肩宽对比:")
        print(f"  YOLO: {yolo_shoulder_width:.1f} mm")
        print(f"  SMPL: {smpl_shoulder_width*1000:.1f} mm")
    
    # 6. 检查坐标系方向
    print(f"\n坐标系方向:")
    print(f"  YOLO: X={yolo_array[0,0]:.0f} (鼻), Y={yolo_array[0,1]:.0f} (向下)")
    print(f"  SMPL: X={smpl_joints[0,0]:.3f} (髋), Y={smpl_joints[0,1]:.3f} (向上)")

if __name__ == "__main__":
    main()
