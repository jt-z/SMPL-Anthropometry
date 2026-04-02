import numpy as np
import torch
import smplx
import os
import warnings
from measure import MeasureBody
from measurement_definitions import STANDARD_LABELS
from scipy.optimize import minimize
import argparse


def get_device():
    if torch.cuda.is_available():
        try:
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
            return torch.device('cuda')
        except Exception as e:
            warnings.warn(f"CUDA可用但初始化失败: {e}")
            warnings.warn("将使用CPU运行")
            return torch.device('cpu')
    else:
        return torch.device('cpu')


class SMPLFitterFromData:
    def __init__(self, model_path="data", model_type="smpl", gender="neutral", device=None):
        self.model_type = model_type
        self.model_path = model_path
        self.gender = gender
        
        if device is None:
            self.device = get_device()
        else:
            self.device = device
        
        if self.device.type == 'cuda':
            print(f"使用设备: CUDA (GPU)")
        else:
            print(f"使用设备: CPU")
        
        self.model = smplx.create(
            model_path=model_path,
            model_type=model_type,
            gender=gender,
            num_betas=10,
            use_face_contour=False,
            ext='pkl'
        ).to(self.device)
        
        self.measurer = MeasureBody(model_type=model_type)
        
        self.coco_to_smpl_mapping = {
            0: 15,   # nose -> head
            5: 16,   # left_shoulder -> left_shoulder
            6: 17,   # right_shoulder -> right_shoulder
            7: 18,   # left_elbow -> left_elbow
            8: 19,   # right_elbow -> right_elbow
            9: 20,   # left_wrist -> left_wrist
            10: 21,  # right_wrist -> right_wrist
            11: 1,   # left_hip -> left_hip
            12: 2,   # right_hip -> right_hip
            13: 4,   # left_knee -> left_knee
            14: 5,   # right_knee -> right_knee
            15: 7,   # left_ankle -> left_ankle
            16: 8,   # right_ankle -> right_ankle
        }
        
        self.smpl_joint_names = {
            0: 'pelvis',
            1: 'left_hip',
            2: 'right_hip',
            3: 'spine1',
            4: 'left_knee',
            5: 'right_knee',
            6: 'spine2',
            7: 'left_ankle',
            8: 'right_ankle',
            9: 'spine3',
            10: 'left_foot',
            11: 'right_foot',
            12: 'neck',
            13: 'left_collar',
            14: 'right_collar',
            15: 'head',
            16: 'left_shoulder',
            17: 'right_shoulder',
            18: 'left_elbow',
            19: 'right_elbow',
            20: 'left_wrist',
            21: 'right_wrist',
            22: 'left_hand',
            23: 'right_hand',
        }
    
    def load_data(self, npz_path):
        data = np.load(npz_path)
        
        keypoints_3d = data['keypoints_3d']
        keypoints_valid = data['keypoints_valid']
        pointcloud = data['pointcloud']
        
        print(f"加载数据成功:")
        print(f"  关键点: {keypoints_3d.shape}")
        print(f"  有效关键点: {np.sum(keypoints_valid)}/{len(keypoints_valid)}")
        print(f"  点云: {pointcloud.shape}")
        
        return keypoints_3d, keypoints_valid, pointcloud
    
    def get_smpl_joints(self, betas, pose):
        if not isinstance(betas, torch.Tensor):
            betas = torch.tensor(betas, dtype=torch.float32)
        if not isinstance(pose, torch.Tensor):
            pose = torch.tensor(pose, dtype=torch.float32)
        
        betas = betas.unsqueeze(0) if betas.dim() == 1 else betas
        pose = pose.unsqueeze(0) if pose.dim() == 1 else pose
        
        betas = betas.to(self.device)
        pose = pose.to(self.device)
        
        with torch.no_grad():
            output = self.model(
                betas=betas,
                body_pose=pose[:, 3:],
                global_orient=pose[:, :3],
                return_verts=True
            )
            
            vertices = output.vertices[0].cpu().numpy()
            joints = output.joints[0].cpu().numpy()
        
        return vertices, joints
    
    def fit_to_keypoints(self, keypoints_3d, keypoints_valid, 
                         initial_betas=None, initial_pose=None,
                         num_iterations=300):
        print("\n开始SMPL关键点拟合...")
        
        target_keypoints = []
        smpl_joint_indices = []
        weights = []
        
        for coco_idx, smpl_idx in self.coco_to_smpl_mapping.items():
            if keypoints_valid[coco_idx]:
                target_keypoints.append(keypoints_3d[coco_idx, :3])
                smpl_joint_indices.append(smpl_idx)
                weights.append(keypoints_3d[coco_idx, 3])
        
        target_keypoints = np.array(target_keypoints)
        weights = np.array(weights)
        
        print(f"使用 {len(target_keypoints)} 个关键点进行拟合")
        
        target_keypoints_m = target_keypoints / 1000.0
        
        if initial_betas is None:
            betas = torch.zeros(10, dtype=torch.float32, requires_grad=True, device=self.device)
        else:
            betas = torch.tensor(initial_betas, dtype=torch.float32, requires_grad=True, device=self.device)
        
        if initial_pose is None:
            pose = torch.zeros(72, dtype=torch.float32, requires_grad=True, device=self.device)
        else:
            pose = torch.tensor(initial_pose, dtype=torch.float32, requires_grad=True, device=self.device)
        
        optimizer = torch.optim.Adam([betas, pose], lr=0.01)
        
        target_torch = torch.tensor(target_keypoints_m, dtype=torch.float32, device=self.device)
        weights_torch = torch.tensor(weights, dtype=torch.float32, device=self.device)
        
        best_loss = float('inf')
        best_betas = betas.detach().clone()
        best_pose = pose.detach().clone()
        
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            
            vertices, joints = self.get_smpl_joints(betas, pose)
            joints_torch = torch.tensor(joints, dtype=torch.float32)
            
            pred_joints = joints_torch[smpl_joint_indices]
            
            diff = (pred_joints - target_torch) ** 2
            weighted_loss = torch.mean(diff * weights_torch.unsqueeze(1))
            
            betas_reg = 0.001 * torch.sum(betas ** 2)
            pose_reg = 0.0001 * torch.sum(pose ** 2)
            
            total_loss = weighted_loss + betas_reg + pose_reg
            
            total_loss.backward()
            optimizer.step()
            
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_betas = betas.detach().clone()
                best_pose = pose.detach().clone()
            
            if iteration % 50 == 0:
                print(f"  迭代 {iteration}: 损失 = {total_loss.item():.6f}")
        
        print(f"拟合完成，最佳损失: {best_loss:.6f}")
        
        return best_betas.numpy(), best_pose.numpy()
    
    def fit_to_pointcloud(self, pointcloud, initial_betas=None, 
                          initial_pose=None, num_iterations=200, 
                          num_samples=2000):
        print("\n开始SMPL点云拟合...")
        
        if len(pointcloud) > num_samples:
            indices = np.random.choice(len(pointcloud), num_samples, replace=False)
            sampled_points = pointcloud[indices]
        else:
            sampled_points = pointcloud
        
        target_points_m = sampled_points / 1000.0
        target_torch = torch.tensor(target_points_m, dtype=torch.float32, device=self.device)
        
        if initial_betas is None:
            betas = torch.zeros(10, dtype=torch.float32, requires_grad=True, device=self.device)
        else:
            betas = torch.tensor(initial_betas, dtype=torch.float32, requires_grad=True, device=self.device)
        
        if initial_pose is None:
            pose = torch.zeros(72, dtype=torch.float32, requires_grad=True, device=self.device)
        else:
            pose = torch.tensor(initial_pose, dtype=torch.float32, requires_grad=True, device=self.device)
        
        optimizer = torch.optim.Adam([betas, pose], lr=0.005)
        
        best_loss = float('inf')
        best_betas = betas.detach().clone()
        best_pose = pose.detach().clone()
        
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            
            vertices, joints = self.get_smpl_joints(betas, pose)
            vertices_torch = torch.tensor(vertices, dtype=torch.float32, device=self.device)
            
            distances = torch.cdist(target_torch, vertices_torch)
            min_distances, _ = torch.min(distances, dim=1)
            
            pointcloud_loss = torch.mean(min_distances ** 2)
            
            betas_reg = 0.001 * torch.sum(betas ** 2)
            pose_reg = 0.0001 * torch.sum(pose ** 2)
            
            total_loss = pointcloud_loss + betas_reg + pose_reg
            
            total_loss.backward()
            optimizer.step()
            
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_betas = betas.detach().clone()
                best_pose = pose.detach().clone()
            
            if iteration % 50 == 0:
                print(f"  迭代 {iteration}: 损失 = {total_loss.item():.6f}")
        
        print(f"点云拟合完成，最佳损失: {best_loss:.6f}")
        
        return best_betas.numpy(), best_pose.numpy()
    
    def measure_body(self, betas):
        print("\n开始身体测量...")
        
        betas_torch = torch.tensor(betas, dtype=torch.float32).unsqueeze(0)
        
        self.measurer.from_body_model(gender=self.gender.upper(), shape=betas_torch)
        
        measurement_names = self.measurer.all_possible_measurements
        self.measurer.measure(measurement_names)
        
        self.measurer.label_measurements(STANDARD_LABELS)
        
        measurements = self.measurer.measurements
        labeled_measurements = self.measurer.labeled_measurements
        
        print(f"完成 {len(measurements)} 项测量")
        
        return measurements, labeled_measurements
    
    def visualize_results(self):
        print("\n生成可视化...")
        self.measurer.visualize()
    
    def save_results(self, output_dir, betas, pose, measurements, labeled_measurements):
        os.makedirs(output_dir, exist_ok=True)
        
        np.savez(
            os.path.join(output_dir, "smpl_params.npz"),
            betas=betas,
            pose=pose
        )
        
        with open(os.path.join(output_dir, "measurements.txt"), 'w') as f:
            f.write("SMPL 身体测量结果\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("SMPL 参数:\n")
            f.write(f"  betas: {betas}\n")
            f.write(f"  pose (前6个值): {pose[:6]}\n\n")
            
            f.write("测量结果 (标准标签):\n")
            f.write("-" * 60 + "\n")
            for label, value in labeled_measurements.items():
                f.write(f"{label:3s}: {value:8.2f} cm\n")
            
            f.write("\n测量结果 (详细名称):\n")
            f.write("-" * 60 + "\n")
            for name, value in measurements.items():
                f.write(f"{name:30s}: {value:8.2f} cm\n")
        
        print(f"\n结果已保存到: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='从数据拟合SMPL模型并进行身体测量')
    parser.add_argument('--input', type=str, 
                        default='/home/zjt/dev/On_Git_Projects/3D-Human-Measure/demo3_back_test/output_v2_yolov8-seg(segmenters_yolov8n-seg.pt)/smpl_input.npz',
                        help='输入npz文件路径')
    parser.add_argument('--output', type=str, 
                        default='./output_smpl_fit',
                        help='输出目录')
    parser.add_argument('--model_type', type=str, default='smpl',
                        choices=['smpl', 'smplx'],
                        help='模型类型')
    parser.add_argument('--gender', type=str, default='neutral',
                        choices=['male', 'female', 'neutral'],
                        help='性别')
    parser.add_argument('--model_path', type=str, default='data',
                        help='SMPL模型路径')
    parser.add_argument('--keypoint_iterations', type=int, default=300,
                        help='关键点拟合迭代次数')
    parser.add_argument('--pointcloud_iterations', type=int, default=200,
                        help='点云拟合迭代次数')
    parser.add_argument('--visualize', action='store_true',
                        help='是否可视化结果')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='计算设备 (auto=自动选择, cpu=强制CPU, cuda=强制GPU)')
    
    args = parser.parse_args()
    
    if args.device == 'auto':
        device = None
    elif args.device == 'cpu':
        device = torch.device('cpu')
    elif args.device == 'cuda':
        device = torch.device('cuda')
    
    print("=" * 60)
    print("SMPL 身体模型拟合与测量")
    print("=" * 60)
    print(f"输入文件: {args.input}")
    print(f"输出目录: {args.output}")
    print(f"模型类型: {args.model_type}")
    print(f"性别: {args.gender}")
    print("=" * 60)
    
    fitter = SMPLFitterFromData(
        model_path=args.model_path,
        model_type=args.model_type,
        gender=args.gender,
        device=device
    )
    
    keypoints_3d, keypoints_valid, pointcloud = fitter.load_data(args.input)
    
    betas_kp, pose_kp = fitter.fit_to_keypoints(
        keypoints_3d, keypoints_valid,
        num_iterations=args.keypoint_iterations
    )
    
    betas_final, pose_final = fitter.fit_to_pointcloud(
        pointcloud,
        initial_betas=betas_kp,
        initial_pose=pose_kp,
        num_iterations=args.pointcloud_iterations
    )
    
    measurements, labeled_measurements = fitter.measure_body(betas_final)
    
    print("\n" + "=" * 60)
    print("测量结果 (标准标签)")
    print("=" * 60)
    for label, value in labeled_measurements.items():
        print(f"{label:3s}: {value:8.2f} cm")
    
    print("\n" + "=" * 60)
    print("主要测量结果")
    print("=" * 60)
    important_measurements = ['height', 'chest_circumference', 'waist_circumference', 
                             'hip_circumference', 'shoulder_breadth']
    for name in important_measurements:
        if name in measurements:
            print(f"{name:30s}: {measurements[name]:8.2f} cm")
    
    fitter.save_results(args.output, betas_final, pose_final, 
                       measurements, labeled_measurements)
    
    if args.visualize:
        fitter.visualize_results()
    
    print("\n处理完成！")


if __name__ == "__main__":
    main()
