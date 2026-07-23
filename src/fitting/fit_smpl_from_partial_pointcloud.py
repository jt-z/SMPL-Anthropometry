#!/usr/bin/env python3
"""
部分点云SMPL拟合算法（改进版）
专门针对不完整点云数据的拟合
"""

import numpy as np
import torch
import smplx
import os
import warnings
from scipy.spatial import cKDTree
import argparse


def get_device():
    if torch.cuda.is_available():
        try:
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
            return torch.device('cuda')
        except Exception as e:
            warnings.warn(f"CUDA可用但初始化失败: {e}")
            return torch.device('cpu')
    else:
        return torch.device('cpu')


class PartialPointCloudSMPLFitter:
    """针对部分点云的SMPL拟合器"""
    
    def __init__(self, model_path="data", model_type="smpl", gender="neutral", device=None):
        self.model_type = model_type
        self.model_path = model_path
        self.gender = gender
        
        if device is None:
            self.device = get_device()
        else:
            self.device = device
        
        print(f"使用设备: {self.device}")
        
        # 加载SMPL模型
        self.model = smplx.create(
            model_path=self.model_path,
            model_type=self.model_type,
            gender=self.gender,
            use_face_contour=False,
            num_betas=10,
            ext='pkl'
        ).to(self.device)
    
    def load_pointcloud(self, file_path):
        """加载点云"""
        import trimesh
        
        print(f"\n加载点云: {file_path}")
        mesh = trimesh.load(file_path)
        pointcloud = np.array(mesh.vertices, dtype=np.float32)
        
        print(f"  点云顶点数: {len(pointcloud)}")
        print(f"  范围: X[{pointcloud[:, 0].min():.3f}, {pointcloud[:, 0].max():.3f}]")
        print(f"        Y[{pointcloud[:, 1].min():.3f}, {pointcloud[:, 1].max():.3f}]")
        print(f"        Z[{pointcloud[:, 2].min():.3f}, {pointcloud[:, 2].max():.3f}]")
        
        return pointcloud
    
    def align_pointcloud_to_origin(self, pointcloud):
        """将点云居中到原点"""
        center = pointcloud.mean(axis=0)
        aligned = pointcloud - center
        
        print(f"\n点云对齐:")
        print(f"  原始中心: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
        print(f"  已移动到原点")
        
        return aligned, center
    
    def initial_alignment(self, pointcloud, smpl_vertices):
        """初始对齐（简单的尺度匹配）"""
        print("\n初始对齐...")
        
        # 计算点云的尺度
        pc_scale = np.std(pointcloud)
        smpl_scale = np.std(smpl_vertices)
        
        scale_factor = pc_scale / smpl_scale
        
        print(f"  点云尺度: {pc_scale:.3f}")
        print(f"  SMPL尺度: {smpl_scale:.3f}")
        print(f"  缩放因子: {scale_factor:.3f}")
        
        # 缩放SMPL
        aligned_smpl = smpl_vertices * scale_factor
        
        return aligned_smpl, scale_factor
    
    def fit_to_partial_pointcloud(self, pointcloud, num_iterations=500, lr=0.01):
        """
        拟合SMPL到部分点云
        
        两阶段优化：
        1. 优化全局位置、旋转和缩放
        2. 优化shape参数（betas）
        """
        
        print("\n" + "=" * 70)
        print("开始部分点云SMPL拟合")
        print("=" * 70)
        
        # 1. 将点云居中
        pointcloud_centered, pc_center = self.align_pointcloud_to_origin(pointcloud)
        
        # 2. 转换为tensor
        target_points = torch.tensor(pointcloud_centered, dtype=torch.float32).to(self.device)
        
        # 3. 初始化参数
        betas = torch.zeros(10, dtype=torch.float32, requires_grad=True, device=self.device)
        global_orient = torch.zeros(3, dtype=torch.float32, requires_grad=True, device=self.device)
        transl = torch.zeros(3, dtype=torch.float32, requires_grad=True, device=self.device)
        scale = torch.ones(1, dtype=torch.float32, requires_grad=True, device=self.device)
        
        # 4. 阶段1：优化位置、旋转和缩放（固定shape）
        print("\n阶段1: 优化位置、旋转和缩放...")
        optimizer = torch.optim.Adam([global_orient, transl, scale], lr=lr)
        
        for iteration in range(200):
            optimizer.zero_grad()
            
            # 生成SMPL顶点
            output = self.model(
                betas=betas.unsqueeze(0),
                global_orient=global_orient.unsqueeze(0),
                return_verts=True
            )
            
            smpl_vertices = output.vertices[0]
            
            # 应用缩放和平移
            smpl_vertices = smpl_vertices * scale + transl
            
            # 计算Chamfer距离（双向）
            # 从点云到SMPL
            tree = cKDTree(smpl_vertices.detach().cpu().numpy())
            distances_pc_to_smpl, _ = tree.query(target_points.cpu().numpy())
            loss_pc_to_smpl = torch.tensor(distances_pc_to_smpl, device=self.device).mean()
            
            # 从SMPL到点云
            tree = cKDTree(target_points.cpu().numpy())
            distances_smpl_to_pc, _ = tree.query(smpl_vertices.detach().cpu().numpy())
            loss_smpl_to_pc = torch.tensor(distances_smpl_to_pc, device=self.device).mean()
            
            # 总损失
            chamfer_loss = loss_pc_to_smpl + loss_smpl_to_pc
            
            # 正则化
            scale_reg = 0.1 * (scale - 1.0) ** 2
            rotation_reg = 0.01 * torch.sum(global_orient ** 2)
            
            total_loss = chamfer_loss + scale_reg + rotation_reg
            
            total_loss.backward()
            optimizer.step()
            
            if iteration % 50 == 0:
                print(f"  迭代 {iteration}: 损失={total_loss.item():.6f}, "
                      f"Chamfer={chamfer_loss.item():.6f}, "
                      f"缩放={scale.item():.3f}")
        
        print(f"阶段1完成! 最终缩放={scale.item():.3f}")
        
        # 5. 阶段2：优化shape参数
        print("\n阶段2: 优化shape参数...")
        betas.requires_grad = True
        optimizer = torch.optim.Adam([betas], lr=lr * 0.1)
        
        best_loss = float('inf')
        best_betas = betas.detach().clone()
        
        for iteration in range(300):
            optimizer.zero_grad()
            
            # 生成SMPL顶点
            output = self.model(
                betas=betas.unsqueeze(0),
                global_orient=global_orient.unsqueeze(0),
                return_verts=True
            )
            
            smpl_vertices = output.vertices[0]
            smpl_vertices = smpl_vertices * scale + transl
            
            # 计算Chamfer距离
            tree = cKDTree(smpl_vertices.detach().cpu().numpy())
            distances_pc_to_smpl, _ = tree.query(target_points.cpu().numpy())
            loss_pc_to_smpl = torch.tensor(distances_pc_to_smpl, device=self.device).mean()
            
            tree = cKDTree(target_points.cpu().numpy())
            distances_smpl_to_pc, _ = tree.query(smpl_vertices.detach().cpu().numpy())
            loss_smpl_to_pc = torch.tensor(distances_smpl_to_pc, device=self.device).mean()
            
            chamfer_loss = loss_pc_to_smpl + loss_smpl_to_pc
            
            # Shape正则化
            betas_reg = 0.001 * torch.sum(betas ** 2)
            
            total_loss = chamfer_loss + betas_reg
            
            total_loss.backward()
            optimizer.step()
            
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_betas = betas.detach().clone()
            
            if iteration % 50 == 0:
                print(f"  迭代 {iteration}: 损失={total_loss.item():.6f}, "
                      f"Chamfer={chamfer_loss.item():.6f}")
        
        print(f"\n拟合完成!")
        print(f"  最佳损失: {best_loss:.6f}")
        print(f"  Betas: {best_betas.cpu().numpy()}")
        print(f"  缩放: {scale.item():.3f}")
        print(f"  平移: {transl.detach().cpu().numpy()}")
        
        # 6. 返回最终参数（转换回原始坐标系）
        final_transl = transl.detach().cpu().numpy() + pc_center
        
        return (best_betas.cpu().numpy(), 
                global_orient.detach().cpu().numpy(),
                final_transl,
                scale.item())
    
    def generate_final_mesh(self, betas, global_orient, transl, scale):
        """生成最终的SMPL mesh"""
        with torch.no_grad():
            betas_t = torch.tensor(betas, dtype=torch.float32).unsqueeze(0).to(self.device)
            orient_t = torch.tensor(global_orient, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            output = self.model(
                betas=betas_t,
                global_orient=orient_t,
                return_verts=True
            )
            
            vertices = output.vertices[0].cpu().numpy()
            vertices = vertices * scale + transl
            
            return vertices
    
    def measure_body(self, betas):
        """身体测量"""
        from src.core.measure import MeasureBody
        from src.core.measurement_definitions import STANDARD_LABELS
        
        print("\n开始身体测量...")
        
        betas_torch = torch.tensor(betas, dtype=torch.float32).unsqueeze(0)
        
        measurer = MeasureBody()
        measurer.from_body_model(gender=self.gender.upper(), shape=betas_torch)
        
        measurement_names = measurer.all_possible_measurements
        measurer.measure(measurement_names)
        
        measurements = {}
        for name in measurement_names:
            try:
                value = measurer.values[name]
                measurements[name] = value
            except:
                pass
        
        labeled_measurements = {}
        for label, name in STANDARD_LABELS.items():
            if name in measurements:
                labeled_measurements[label] = measurements[name]
        
        print(f"完成 {len(measurements)} 项测量")
        
        return measurements, labeled_measurements
    
    def save_results(self, output_dir, betas, global_orient, transl, scale, 
                    measurements, labeled_measurements):
        """保存结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存SMPL参数
        params_file = os.path.join(output_dir, 'smpl_params.npz')
        np.savez(params_file,
                 betas=betas,
                 global_orient=global_orient,
                 transl=transl,
                 scale=scale)
        
        # 保存测量结果
        measurements_file = os.path.join(output_dir, 'measurements.txt')
        with open(measurements_file, 'w', encoding='utf-8') as f:
            f.write("SMPL 身体测量结果（部分点云拟合）\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("SMPL 参数:\n")
            f.write(f"  betas: {betas}\n")
            f.write(f"  scale: {scale:.3f}\n")
            f.write(f"  transl: {transl}\n\n")
            
            f.write("测量结果 (标准标签):\n")
            f.write("-" * 60 + "\n")
            for label, value in sorted(labeled_measurements.items()):
                f.write(f"{label:3s}: {value:8.2f} cm\n")
            
            f.write("\n测量结果 (详细名称):\n")
            f.write("-" * 60 + "\n")
            for name, value in sorted(measurements.items())[:10]:
                f.write(f"{name:30s}: {value:8.2f} cm\n")
        
        print(f"\n结果已保存到: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='部分点云SMPL拟合')
    parser.add_argument('--input', type=str, required=True,
                        help='输入点云文件 (.ply)')
    parser.add_argument('--output', type=str, required=True,
                        help='输出目录')
    parser.add_argument('--gender', type=str, default='neutral',
                        choices=['neutral', 'male', 'female'])
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.01)
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("部分点云SMPL拟合（改进版）")
    print("=" * 70)
    print(f"输入文件: {args.input}")
    print(f"输出目录: {args.output}")
    print(f"性别: {args.gender}")
    print("=" * 70)
    
    # 1. 创建拟合器
    device = get_device()
    fitter = PartialPointCloudSMPLFitter(
        model_path="data",
        model_type="smpl",
        gender=args.gender,
        device=device
    )
    
    # 2. 加载点云
    pointcloud = fitter.load_pointcloud(args.input)
    
    # 3. 拟合
    betas, global_orient, transl, scale = fitter.fit_to_partial_pointcloud(
        pointcloud,
        num_iterations=args.iterations,
        lr=args.lr
    )
    
    # 4. 测量
    measurements, labeled_measurements = fitter.measure_body(betas)
    
    # 5. 显示主要结果
    print("\n" + "=" * 70)
    print("主要测量结果")
    print("=" * 70)
    important = ['height', 'chest_circumference', 'waist_circumference', 
                 'hip_circumference', 'shoulder_breadth']
    for name in important:
        if name in measurements:
            print(f"{name:30s}: {measurements[name]:8.2f} cm")
    
    # 6. 保存结果
    fitter.save_results(args.output, betas, global_orient, transl, scale,
                        measurements, labeled_measurements)
    
    print("\n处理完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
