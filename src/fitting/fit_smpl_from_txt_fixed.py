#!/usr/bin/env python3
"""
从TXT测量结果文件进行SMPL拟合（修复版）
修复问题：
1. 单位转换 (mm -> m)
2. 坐标系对齐 (YOLO向下Y轴 -> SMPL向上Y轴)
3. Procrustes初始对齐
4. 优化器和损失函数改进
"""

import os
import numpy as np
import torch
import smplx
import re
import warnings
from scipy.spatial.transform import Rotation as R

def procrustes_align(source_points, target_points):
    """
    使用Procrustes分析进行刚性对齐
    
    Args:
        source_points: (N, 3) numpy array
        target_points: (N, 3) numpy array
        
    Returns:
        R_mat: 旋转矩阵 (3, 3)
        t: 平移向量 (3,)
        scale: 缩放因子
    """
    assert source_points.shape == target_points.shape
    
    # 中心化
    source_center = np.mean(source_points, axis=0)
    target_center = np.mean(target_points, axis=0)
    
    source_centered = source_points - source_center
    target_centered = target_points - target_center
    
    # 计算协方差矩阵
    H = source_centered.T @ target_centered
    
    # SVD分解
    U, S, Vt = np.linalg.svd(H)
    R_mat = Vt.T @ U.T
    
    # 确保是右手坐标系
    if np.linalg.det(R_mat) < 0:
        Vt[-1, :] *= -1
        R_mat = Vt.T @ U.T
    
    # 计算缩放
    source_scale = np.sum(source_centered ** 2)
    target_scale = np.sum(target_centered ** 2)
    scale = np.sqrt(target_scale / source_scale) if source_scale > 0 else 1.0
    
    # 计算平移
    t = target_center - scale * (R_mat @ source_center)
    
    return R_mat, t, scale

def get_device():
    """选择设备"""
    if torch.cuda.is_available():
        try:
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
            return torch.device('cuda')
        except Exception as e:
            warnings.warn(f"CUDA available but initialization failed: {e}")
            return torch.device('cpu')
    return torch.device('cpu')

class TXTMeasurementLoader:
    def __init__(self):
        self.coco_keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        self.camera_params = {
            'fx': 505.440063,
            'fy': 505.451843,
            'cx': 326.825256,
            'cy': 335.328552
        }
    
    def parse_txt_file(self, txt_path):
        """解析TXT文件，返回标准化到SMPL坐标系的关键点"""
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        keypoints_3d_dict = {}
        
        for kp_name in self.coco_keypoint_names:
            pattern = rf'{kp_name}\s+:.+?3D=\[\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\]'
            match = re.search(pattern, content)
            
            if match:
                x = float(match.group(1))
                y = float(match.group(2))
                z = float(match.group(3))
                keypoints_3d_dict[kp_name] = np.array([x, y, z])
        
        # 转换为数组 (17, 3)，单位：mm
        keypoints_3d_mm = np.array([
            keypoints_3d_dict.get(name, [0, 0, 0]) 
            for name in self.coco_keypoint_names
        ], dtype=np.float32)
        
        confidences = np.array([1.0 if name in keypoints_3d_dict else 0.0 
                                for name in self.coco_keypoint_names], 
                               dtype=np.float32)
        
        # ============================================
        # 关键：坐标系转换（YOLO相机坐标 -> SMPL坐标）
        # ============================================
        
        # 1. 单位转换: mm -> m
        keypoints_3d_m = keypoints_3d_mm / 1000.0
        
        # 2. 坐标系翻转：YOLO的Y轴向下增大，SMPL的Y轴向上增大
        # 注意：我们先暂时不翻转，Procrustes会自动处理
        # 但我们需要确保数据集中化
        
        # 3. 以髋部为中心（SMPL原点在骨盆）
        left_hip_idx = self.coco_keypoint_names.index('left_hip')
        right_hip_idx = self.coco_keypoint_names.index('right_hip')
        
        if confidences[left_hip_idx] > 0 and confidences[right_hip_idx] > 0:
            hip_center = (keypoints_3d_m[left_hip_idx] + keypoints_3d_m[right_hip_idx]) / 2
        else:
            hip_center = np.mean(keypoints_3d_m[confidences > 0], axis=0)
        
        # 中心化到髋部
        keypoints_centered = keypoints_3d_m - hip_center
        
        # 4. 翻转Y轴（YOLO向下 -> SMPL向上）
        keypoints_centered[:, 1] = -keypoints_centered[:, 1]
        
        # 生成点云（关键点骨架）
        pointcloud = self.generate_pointcloud_from_keypoints(keypoints_centered)
        
        keypoints_with_conf = np.concatenate(
            [keypoints_centered, confidences.reshape(-1, 1)], 
            axis=1
        )
        
        return keypoints_with_conf, confidences > 0, pointcloud
    
    def generate_pointcloud_from_keypoints(self, keypoints_3d, num_points_per_segment=50):
        """从关键点生成模拟点云"""
        segments = [
            (0, 5), (0, 6),  # nose to shoulders
            (5, 6),           # shoulders
            (5, 7), (6, 8),   # shoulders to elbows
            (7, 9), (8, 10),  # elbows to wrists
            (5, 11), (6, 12), # shoulders to hips
            (11, 12),          # hips
            (11, 13), (12, 14), # hips to knees
            (13, 15), (14, 16)  # knees to ankles
        ]
        
        points = []
        for i, j in segments:
            for t in np.linspace(0, 1, num_points_per_segment):
                p = keypoints_3d[i] * (1-t) + keypoints_3d[j] * t
                points.append(p)
        
        return np.array(points)

class SMPLFitterFromMeasurements:
    def __init__(self, model_path="data", model_type="smpl", gender="neutral", device=None):
        self.model_type = model_type
        self.model_path = model_path
        self.gender = gender
        
        if device is None:
            self.device = get_device()
        else:
            self.device = device
        
        print(f"Device: {'CUDA (GPU)' if self.device.type == 'cuda' else 'CPU'}")
        
        self.model = smplx.create(
            model_path=model_path,
            model_type=model_type,
            gender=gender,
            num_betas=10,
            use_face_contour=False,
            ext='pkl'
        ).to(self.device)
        
        # COCO → SMPL关键点映射（索引）
        self.coco_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # SMPL 关节索引到名称的映射
        self.smpl_joint_names = [
            'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 
            'right_knee', 'spine2', 'left_ankle', 'right_ankle', 'spine3', 
            'left_foot', 'right_foot', 'neck', 'left_shoulder', 'right_shoulder', 
            'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hand', 
            'right_hand', 'jaw', 'left_eye', 'right_eye'
        ]
        
        # 构建 COCO名称 → SMPL关节索引 的映射
        self.coco_to_smpl = {}
        for smpl_idx, name in enumerate(self.smpl_joint_names):
            if name in self.coco_names:
                self.coco_to_smpl[name] = smpl_idx
        
        # 同时，也用列表形式
        self.smpl_joint_indices_for_coco = []
        for coco_name in self.coco_names:
            if coco_name in self.coco_to_smpl:
                self.smpl_joint_indices_for_coco.append(self.coco_to_smpl[coco_name])
            else:
                self.smpl_joint_indices_for_coco.append(-1)  # 无效标记
        
        self.measurer = None
    
    def get_smpl_joints(self, betas, pose=None):
        """前向传播获取SMPL关节和顶点"""
        if pose is None:
            pose = torch.zeros(72, device=self.device)
        
        betas_torch = torch.tensor(betas, dtype=torch.float32, device=self.device).unsqueeze(0)
        pose_torch = pose.reshape(1, 72)
        
        output = self.model(
            betas=betas_torch,
            body_pose=pose_torch[:, 3:],
            global_orient=pose_torch[:, :3],
            transl=torch.zeros(1, 3, device=self.device),
            return_verts=True,
            return_joints=True
        )
        
        vertices = output.vertices.detach().cpu().numpy()[0]
        joints = output.joints.detach().cpu().numpy()[0]
        
        return vertices, joints
    
    def fit_to_keypoints(self, keypoints_with_conf, keypoints_valid, 
                          num_iterations=500, lr=0.01):
        """
        拟合SMPL到关键点（修复版）
        使用Procrustes初始对齐，然后优化betas和pose
        """
        print("\nStep 1: Keypoint fitting with Procrustes initial alignment")
        
        # 提取有效关键点
        valid_indices = np.where(keypoints_valid)[0]
        
        # 同时存在于COCO和SMPL的关键点
        common_valid = []
        source_kps = []
        for coco_idx in valid_indices:
            smpl_idx = self.smpl_joint_indices_for_coco[coco_idx]
            if smpl_idx >= 0:
                common_valid.append((coco_idx, smpl_idx))
                source_kps.append(keypoints_with_conf[coco_idx, :3])
        
        if len(common_valid) < 4:
            print("Error: Not enough common keypoints for Procrustes!")
            return np.zeros(10), np.zeros(72)
        
        source_kps = np.array(source_kps)
        
        # 1. 先获取SMPL零参数关节作为目标
        _, smpl_joints_zero = self.get_smpl_joints(np.zeros(10))
        
        target_kps = np.array([smpl_joints_zero[smpl_idx] for _, smpl_idx in common_valid])
        
        print(f"Using {len(common_valid)} common keypoints for Procrustes")
        
        # 2. Procrustes对齐：找到从SMPL零参数到输入关键点的变换
        # 注意：我们要反过来，因为我们要把输入关键点对齐到SMPL空间
        R_mat, t, scale = procrustes_align(target_kps, source_kps)
        
        print(f"Procrustes scale: {scale:.3f}")
        print(f"Procrustes translation: {t}")
        
        # 3. 初始化优化变量
        # 注意：我们现在不优化全局变换，因为Procrustes已经处理了
        # 我们只优化betas（形状）和body_pose（姿态）
        
        betas = torch.zeros(10, dtype=torch.float32, requires_grad=True, device=self.device)
        pose = torch.zeros(72, dtype=torch.float32, requires_grad=True, device=self.device)
        
        # 使用Adam优化器
        optimizer = torch.optim.Adam([betas, pose], lr=lr)
        
        best_loss = float('inf')
        best_betas = betas.detach().clone()
        best_pose = pose.detach().clone()
        
        print("\nStarting optimization...")
        
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            
            # 前向传播
            betas_torch = betas.unsqueeze(0)
            pose_torch = pose.reshape(1, 72)
            
            output = self.model(
                betas=betas_torch,
                body_pose=pose_torch[:, 3:],
                global_orient=pose_torch[:, :3],
                transl=torch.zeros(1, 3, device=self.device),
                return_verts=True,
                return_joints=True
            )
            
            pred_joints = output.joints[0]
            
            # 计算损失
            loss = 0.0
            for i, (coco_idx, smpl_idx) in enumerate(common_valid):
                # 输入关键点
                target_pt = torch.tensor(source_kps[i], dtype=torch.float32, device=self.device)
                
                # SMPL预测关键点
                pred_pt = pred_joints[smpl_idx]
                
                # 应用Procrustes变换到SMPL预测点
                # (实际上我们反过来：把SMPL点变换到输入空间)
                R_torch = torch.tensor(R_mat.T, dtype=torch.float32, device=self.device)
                t_torch = torch.tensor(t, dtype=torch.float32, device=self.device)
                
                pred_transformed = scale * (pred_pt @ R_torch) + t_torch
                
                loss += torch.sum((pred_transformed - target_pt) ** 2)
            
            # 正则化
            betas_reg = 1e-4 * torch.sum(betas ** 2)
            pose_reg = 1e-6 * torch.sum(pose ** 2)
            
            total_loss = loss + betas_reg + pose_reg
            
            total_loss.backward()
            optimizer.step()
            
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_betas = betas.detach().clone()
                best_pose = pose.detach().clone()
            
            if iteration % 100 == 0:
                print(f"  Iteration {iteration}: Loss = {total_loss.item():.6f}")
        
        print(f"Keypoint fitting complete, best loss: {best_loss:.6f}")
        
        return best_betas.cpu().numpy(), best_pose.cpu().numpy()
    
    def measure_body(self, betas):
        """测量身体（完全复用原代码）"""
        from src.core.measure import MeasureBody
        from src.core.measurement_definitions import STANDARD_LABELS
        
        print("\nStarting body measurements...")
        
        if self.measurer is None:
            self.measurer = MeasureBody(model_type=self.model_type)
        
        betas_torch = torch.tensor(betas, dtype=torch.float32).unsqueeze(0)
        
        self.measurer.from_body_model(gender=self.gender.upper(), shape=betas_torch)
        
        measurement_names = self.measurer.all_possible_measurements
        self.measurer.measure(measurement_names)
        
        self.measurer.label_measurements(STANDARD_LABELS)
        
        print(f"Completed {len(self.measurer.measurements)} measurements")
        
        return self.measurer.measurements, self.measurer.labeled_measurements
    
    def visualize_results(self):
        """可视化结果（复用原代码）"""
        if self.measurer:
            print("\nGenerating visualization...")
            self.measurer.visualize()
        else:
            print("No measurer available, run measure_body first.")
    
    def save_results(self, output_dir, betas, measurements, labeled):
        """保存结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        np.savez(os.path.join(output_dir, 'smpl_params.npz'), 
                betas=betas, 
                model_type=self.model_type,
                gender=self.gender)
        
        with open(os.path.join(output_dir, 'measurements.txt'), 'w', encoding='utf-8') as f:
            f.write("SMPL 身体测量结果\n")
            f.write("="*60 + "\n\n")
            f.write(f"SMPL 参数:\n")
            f.write(f"  betas: {betas}\n\n")
            f.write("测量结果 (标准标签):\n")
            f.write("-"*60 + "\n")
            for label, val in sorted(labeled.items()):
                f.write(f"{label:3s} : {val:8.2f} cm\n")
            f.write("\n测量结果 (详细名称):\n")
            f.write("-"*60 + "\n")
            for name, val in measurements.items():
                f.write(f"{name:30s} : {val:8.2f} cm\n")
        
        print(f"\nResults saved to: {output_dir}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="SMPL Fitting from TXT Measurements (Fixed)")
    parser.add_argument('--input', type=str, 
                        default="/home/zjt/dev/On_Git_Projects/3D-Human-Measure/frame_output/frame_1860_yolo_measure_results.txt",
                        help="Input TXT file path")
    parser.add_argument('--output', type=str, default="./output_from_txt_fixed", 
                        help="Output directory")
    parser.add_argument('--model_type', type=str, default="smpl", 
                        choices=["smpl", "smplx"], help="Model type")
    parser.add_argument('--gender', type=str, default="neutral", 
                        choices=["male", "female", "neutral"], help="Gender")
    parser.add_argument('--keypoint_iterations', type=int, default=500,
                        help="Keypoint fitting iterations")
    parser.add_argument('--lr', type=float, default=0.01,
                        help="Learning rate")
    parser.add_argument('--visualize', action='store_true', 
                        help="Enable visualization")
    parser.add_argument('--device', type=str, default="auto",
                        choices=["auto", "cpu", "cuda"], help="Device")
    
    args = parser.parse_args()
    
    print("="*60)
    print("SMPL Model Fitting from TXT Measurements (Fixed)")
    print("="*60)
    print(f"Input file: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Model type: {args.model_type}")
    print(f"Gender: {args.gender}")
    print("="*60)
    
    if args.device == "cpu":
        device = torch.device('cpu')
    elif args.device == "cuda":
        device = torch.device('cuda')
    else:
        device = None
    
    loader = TXTMeasurementLoader()
    keypoints_with_conf, keypoints_valid, pointcloud = loader.parse_txt_file(args.input)
    
    print(f"\nData loaded:")
    print(f"  Keypoints: {keypoints_with_conf.shape}")
    print(f"  Valid keypoints: {np.sum(keypoints_valid)}/{len(keypoints_valid)}")
    print(f"  Pointcloud: {pointcloud.shape}")
    
    fitter = SMPLFitterFromMeasurements(
        model_path="data", 
        model_type=args.model_type, 
        gender=args.gender,
        device=device
    )
    
    betas, pose = fitter.fit_to_keypoints(
        keypoints_with_conf, 
        keypoints_valid, 
        num_iterations=args.keypoint_iterations,
        lr=args.lr
    )
    
    print(f"\nOptimized betas: {betas}")
    
    print("\nStep 2: Body measurements")
    measurements, labeled = fitter.measure_body(betas)
    
    print(f"\n{'='*60}")
    print("Measurement Results (Standard Labels)")
    print(f"{'='*60}")
    for label, val in sorted(labeled.items()):
        print(f"{label:3s} : {val:8.2f} cm")
    
    fitter.save_results(args.output, betas, measurements, labeled)
    
    if args.visualize:
        fitter.visualize_results()
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main()
