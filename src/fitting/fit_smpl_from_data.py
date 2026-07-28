import numpy as np
import torch
import smplx
import os
import warnings
from src.core.measure import MeasureBody
from src.core.measurement_definitions import STANDARD_LABELS
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

        print(f"已加载SMPL模型: 类型={model_type}, 性别={gender}")
        
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
    
    def load_data(self, input_path):
        """加载数据，支持NPZ和PLY/OBJ格式"""
        import os

        # 获取文件扩展名
        _, ext = os.path.splitext(input_path)
        ext = ext.lower()

        if ext == '.npz':
            # 原有逻辑：加载NPZ格式（关键点数据）
            print(f"检测到NPZ格式，加载关键点数据...")
            data = np.load(input_path)

            keypoints_3d = data['keypoints_3d']
            keypoints_valid = data['keypoints_valid']
            pointcloud = data['pointcloud']

            print(f"加载数据成功:")
            print(f"  关键点: {keypoints_3d.shape}")
            print(f"  有效关键点: {np.sum(keypoints_valid)}/{len(keypoints_valid)}")
            print(f"  点云: {pointcloud.shape}")

        elif ext in ['.ply', '.obj']:
            # 新逻辑：加载PLY/OBJ格式（点云/网格数据）
            print(f"检测到{ext.upper()}格式，加载点云/网格数据...")

            try:
                import trimesh
            except ImportError:
                raise ImportError("需要安装trimesh库来读取PLY/OBJ文件: pip install trimesh")

            # 加载网格
            mesh = trimesh.load(input_path)

            # 提取顶点作为点云
            if hasattr(mesh, 'vertices'):
                pointcloud = np.array(mesh.vertices)
            else:
                raise ValueError(f"无法从{input_path}提取顶点数据")

            print(f"加载数据成功:")
            print(f"  顶点数: {len(pointcloud)}")
            print(f"  坐标范围:")
            print(f"    X: [{pointcloud[:, 0].min():.3f}, {pointcloud[:, 0].max():.3f}]")
            print(f"    Y: [{pointcloud[:, 1].min():.3f}, {pointcloud[:, 1].max():.3f}]")
            print(f"    Z: [{pointcloud[:, 2].min():.3f}, {pointcloud[:, 2].max():.3f}]")

            # 点云数据没有关键点信息
            keypoints_3d = None
            keypoints_valid = None

        else:
            raise ValueError(f"不支持的文件格式: {ext}。支持的格式: .npz, .ply, .obj")

        return keypoints_3d, keypoints_valid, pointcloud

    def transform_coordinate_system(self, pointcloud, transform_type='flip_y'):
        """
        转换点云坐标系以匹配SMPL坐标系

        SMPL坐标系：X=左右(右+), Y=上下(上+), Z=前后(前+)

        参数:
            pointcloud: 输入点云 (N, 3)
            transform_type: 转换类型
                'flip_y': 翻转Y轴 (Y -> -Y)，适用于Y轴向下为正的坐标系
                'flip_z': 翻转Z轴 (Z -> -Z)
                'swap_yz': 交换Y和Z轴
                'swap_yz_flip': 交换Y和Z轴，并翻转新的Y轴
                'none': 不进行转换
        """
        transformed = pointcloud.copy()

        if transform_type == 'flip_y':
            transformed[:, 1] = -transformed[:, 1]
            print(f"  坐标系转换: 翻转Y轴 (Y -> -Y)")
        elif transform_type == 'flip_z':
            transformed[:, 2] = -transformed[:, 2]
            print(f"  坐标系转换: 翻转Z轴 (Z -> -Z)")
        elif transform_type == 'swap_yz':
            transformed[:, [1, 2]] = transformed[:, [2, 1]]
            print(f"  坐标系转换: 交换Y和Z轴")
        elif transform_type == 'swap_yz_flip':
            transformed[:, [1, 2]] = transformed[:, [2, 1]]
            transformed[:, 1] = -transformed[:, 1]
            print(f"  坐标系转换: 交换Y和Z轴，并翻转Y轴")
        elif transform_type == 'none':
            print(f"  坐标系转换: 无")
        else:
            raise ValueError(f"未知的转换类型: {transform_type}")

        return transformed
    
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

        return best_betas.cpu().numpy(), best_pose.cpu().numpy()
    
    def fit_to_pointcloud(self, pointcloud, initial_betas=None,
                          initial_pose=None, num_iterations=200,
                          num_samples=2000, freeze_pose=False,
                          betas_reg_weight=0.001, pose_reg_weight=0.0001,
                          bidirectional_weight=0.5, pose_type='natural',
                          use_symmetry=False, two_stage=False, torso_ratio=0.4):
        print("\n开始SMPL点云拟合...")

        if freeze_pose:
            print("  姿态冻结模式：只优化body shape和位置，姿态保持标准站姿")

        # 两阶段拟合：先用躯干对齐，再用全部点云精细拟合
        if two_stage:
            print(f"  使用两阶段拟合策略")
            print(f"  阶段1: 使用躯干中心部分（Y轴中间{torso_ratio*100:.0f}%）进行粗拟合")

            # 提取躯干部分（Y轴中间部分，排除头部和脚部）
            y_min, y_max = pointcloud[:, 1].min(), pointcloud[:, 1].max()
            y_range = y_max - y_min
            y_center = (y_min + y_max) / 2

            # 躯干范围：中心上下各 torso_ratio/2
            torso_y_min = y_center - y_range * torso_ratio / 2
            torso_y_max = y_center + y_range * torso_ratio / 2

            torso_mask = (pointcloud[:, 1] >= torso_y_min) & (pointcloud[:, 1] <= torso_y_max)
            torso_points = pointcloud[torso_mask]

            print(f"    原始点云: {len(pointcloud)} 点, Y范围: [{y_min:.3f}, {y_max:.3f}]")
            print(f"    躯干点云: {len(torso_points)} 点, Y范围: [{torso_y_min:.3f}, {torso_y_max:.3f}]")

            if len(torso_points) < 100:
                print("  警告: 躯干点云太少，使用全部点云")
                torso_points = pointcloud

            # 阶段1：用躯干粗拟合（获取初始betas和transl）
            print(f"  阶段1: 躯干粗拟合 ({num_iterations//2} 次迭代)...")
            betas_stage1, pose_stage1, transl_stage1 = self._fit_stage(
                torso_points, initial_betas, initial_pose,
                num_iterations=num_iterations//2,
                num_samples=min(num_samples, len(torso_points)),
                freeze_pose=freeze_pose,
                betas_reg_weight=betas_reg_weight,
                pose_reg_weight=pose_reg_weight,
                bidirectional_weight=bidirectional_weight,
                pose_type=pose_type
            )

            # 阶段2：用全部点云精细拟合
            print(f"\n  阶段2: 全点云精细拟合 ({num_iterations} 次迭代)...")
            return self._fit_stage(
                pointcloud, betas_stage1, pose_stage1,
                num_iterations=num_iterations,
                num_samples=num_samples,
                freeze_pose=freeze_pose,
                betas_reg_weight=betas_reg_weight,
                pose_reg_weight=pose_reg_weight,
                bidirectional_weight=bidirectional_weight,
                pose_type=pose_type,
                initial_transl=transl_stage1
            )
        else:
            # 单阶段拟合
            return self._fit_stage(
                pointcloud, initial_betas, initial_pose,
                num_iterations=num_iterations,
                num_samples=num_samples,
                freeze_pose=freeze_pose,
                betas_reg_weight=betas_reg_weight,
                pose_reg_weight=pose_reg_weight,
                bidirectional_weight=bidirectional_weight,
                pose_type=pose_type
            )

    def _fit_stage(self, pointcloud, initial_betas=None, initial_pose=None,
                   num_iterations=200, num_samples=2000, freeze_pose=False,
                   betas_reg_weight=0.001, pose_reg_weight=0.0001,
                   bidirectional_weight=0.5, pose_type='natural',
                   initial_transl=None):

        if len(pointcloud) > num_samples:
            indices = np.random.choice(len(pointcloud), num_samples, replace=False)
            sampled_points = pointcloud[indices]
        else:
            sampled_points = pointcloud

        # 点云已经是米单位，不需要转换
        target_torch = torch.tensor(sampled_points, dtype=torch.float32, device=self.device)

        # 初始化平移参数到点云中心
        pc_center = sampled_points.mean(axis=0)
        transl = torch.tensor(pc_center, dtype=torch.float32, requires_grad=True, device=self.device)

        print(f"  点云中心: {pc_center}")
        print(f"  点云范围: X[{sampled_points[:, 0].min():.3f}, {sampled_points[:, 0].max():.3f}] "
              f"Y[{sampled_points[:, 1].min():.3f}, {sampled_points[:, 1].max():.3f}] "
              f"Z[{sampled_points[:, 2].min():.3f}, {sampled_points[:, 2].max():.3f}]")

        if initial_betas is None:
            betas = torch.zeros(10, dtype=torch.float32, requires_grad=True, device=self.device)
        else:
            betas = torch.tensor(initial_betas, dtype=torch.float32, requires_grad=True, device=self.device)

        if initial_pose is None:
            # SMPL pose参数：前3个是global_orient，后69个是body_pose（23个关节 x 3）
            # 左肩关节(joint 16)对应body_pose[39:42]，右肩关节(joint 17)对应body_pose[42:45]
            pose = torch.zeros(72, dtype=torch.float32, device=self.device)

            if pose_type == 'apose':
                # A-pose: 手臂向外伸展约30-45度
                # 左肩：绕Z轴旋转（向外）
                pose[41] = -0.5236  # -30度 = -π/6 rad，手臂向外
                # 右肩：绕Z轴旋转（向外）
                pose[44] = 0.5236   # 30度，手臂向外
                print(f"  初始姿态: A-pose (手臂向外伸展30度)")
            elif pose_type == 'tpose':
                # T-pose: 手臂完全水平伸展
                pose[41] = -1.5708  # -90度 = -π/2 rad
                pose[44] = 1.5708   # 90度
                print(f"  初始姿态: T-pose (手臂水平伸展)")
            elif pose_type == 'natural':
                # 自然站姿：手臂下垂
                pose[41] = 0.7854   # 45度 = π/4 rad，向内
                pose[44] = -0.7854  # 45度，向内
                print(f"  初始姿态: 自然站姿 (手臂下垂)")
            else:
                # 默认：零姿态（标准站姿）
                print(f"  初始姿态: 零姿态 (标准SMPL姿态)")

            pose.requires_grad = not freeze_pose
        else:
            pose = torch.tensor(initial_pose, dtype=torch.float32, requires_grad=(not freeze_pose), device=self.device)
            print(f"  使用提供的初始姿态")

        # 只优化betas和transl，pose固定为站姿
        if freeze_pose:
            optimizer = torch.optim.Adam([betas, transl], lr=0.01)
        else:
            optimizer = torch.optim.Adam([betas, pose, transl], lr=0.005)

        best_loss = float('inf')
        best_betas = betas.detach().clone()
        best_pose = pose.detach().clone()
        best_transl = transl.detach().clone()

        for iteration in range(num_iterations):
            optimizer.zero_grad()

            # 直接用torch计算，保持梯度
            betas_batch = betas.unsqueeze(0)
            pose_batch = pose.unsqueeze(0)

            output = self.model(
                betas=betas_batch,
                body_pose=pose_batch[:, 3:],
                global_orient=pose_batch[:, :3],
                return_verts=True
            )

            vertices_torch = output.vertices[0] + transl  # 加上平移

            # 点云到模型的距离（保证点云上的点都靠近模型）
            distances_pc_to_model = torch.cdist(target_torch, vertices_torch)
            min_distances_pc_to_model, _ = torch.min(distances_pc_to_model, dim=1)
            pc_to_model_loss = torch.mean(min_distances_pc_to_model ** 2)

            # 模型到点云的距离（防止模型顶点飘离点云，尤其对不完整点云重要）
            # 但权重要低一些，因为模型的正面没有对应的点云数据
            distances_model_to_pc = torch.cdist(vertices_torch, target_torch)
            min_distances_model_to_pc, _ = torch.min(distances_model_to_pc, dim=1)
            model_to_pc_loss = torch.mean(min_distances_model_to_pc ** 2)

            # 组合双向损失，点云→模型的权重更高
            pointcloud_loss = pc_to_model_loss + bidirectional_weight * model_to_pc_loss

            betas_reg = betas_reg_weight * torch.sum(betas ** 2)
            pose_reg = pose_reg_weight * torch.sum(pose ** 2)

            total_loss = pointcloud_loss + betas_reg + pose_reg

            total_loss.backward()
            optimizer.step()

            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_betas = betas.detach().clone()
                best_pose = pose.detach().clone()
                best_transl = transl.detach().clone()

            if iteration % 50 == 0:
                print(f"  迭代 {iteration}: 损失 = {total_loss.item():.6f}, 平移 = {transl.detach().cpu().numpy()}")

        print(f"点云拟合完成，最佳损失: {best_loss:.6f}")
        print(f"最终平移: {best_transl.cpu().numpy()}")

        return best_betas.cpu().numpy(), best_pose.cpu().numpy(), best_transl.cpu().numpy()
    
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
    
    def save_results(self, output_dir, betas, pose, measurements, labeled_measurements, transl=None, pointcloud=None):
        os.makedirs(output_dir, exist_ok=True)

        save_dict = {
            'betas': betas,
            'pose': pose,
            'gender': self.gender  # 保存性别信息
        }
        if transl is not None:
            save_dict['transl'] = transl

        np.savez(
            os.path.join(output_dir, "smpl_params.npz"),
            **save_dict
        )

        # 保存拟合后的SMPL网格模型（使用当前模型的性别）
        print(f"生成SMPL网格 (性别: {self.gender})...")
        betas_torch = torch.tensor(betas, dtype=torch.float32, device=self.device).unsqueeze(0)
        pose_torch = torch.tensor(pose, dtype=torch.float32, device=self.device).unsqueeze(0)

        output = self.model(
            betas=betas_torch,
            body_pose=pose_torch[:, 3:],
            global_orient=pose_torch[:, :3],
            return_verts=True
        )

        vertices = output.vertices[0].detach().cpu().numpy()
        if transl is not None:
            vertices += transl

        faces = self.model.faces

        # 保存为OBJ格式
        obj_path = os.path.join(output_dir, "fitted_smpl.obj")
        with open(obj_path, 'w') as f:
            for v in vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        print(f"已保存SMPL模型: {obj_path}")

        # 保存为PLY格式（带面信息）
        ply_path = os.path.join(output_dir, "fitted_smpl.ply")
        with open(ply_path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write(f"element face {len(faces)}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            for v in vertices:
                f.write(f"{v[0]} {v[1]} {v[2]}\n")
            for face in faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
        print(f"已保存SMPL模型: {ply_path}")

        # 如果提供了原始点云，保存它以便对比
        if pointcloud is not None:
            pc_path = os.path.join(output_dir, "original_pointcloud.ply")
            with open(pc_path, 'w') as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {len(pointcloud)}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("end_header\n")
                for p in pointcloud:
                    f.write(f"{p[0]} {p[1]} {p[2]}\n")
            print(f"已保存原始点云: {pc_path}")

            # 保存SMPL网格的采样点（用于点云对比工具如CloudCompare）
            smpl_sampled_path = os.path.join(output_dir, "smpl_sampled_points.ply")
            # 从SMPL网格表面采样点（取顶点作为采样点）
            with open(smpl_sampled_path, 'w') as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {len(vertices)}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
                f.write("end_header\n")
                for v in vertices:
                    # 红色 (255, 100, 100)
                    f.write(f"{v[0]} {v[1]} {v[2]} 255 100 100\n")
            print(f"已保存SMPL采样点云: {smpl_sampled_path} (红色)")

            # 创建叠加可视化（SMPL模型 + 原始点云）
            self._create_overlay_visualization(vertices, faces, pointcloud, output_dir)

        with open(os.path.join(output_dir, "measurements.txt"), 'w') as f:
            f.write("SMPL 身体测量结果\n")
            f.write("=" * 60 + "\n\n")

            f.write("SMPL 参数:\n")
            f.write(f"  betas: {betas}\n")
            f.write(f"  pose (前6个值): {pose[:6]}\n")
            if transl is not None:
                f.write(f"  transl: {transl}\n")
            f.write("\n")

            f.write("测量结果 (标准标签):\n")
            f.write("-" * 60 + "\n")
            for label, value in labeled_measurements.items():
                f.write(f"{label:3s}: {value:8.2f} cm\n")

            f.write("\n测量结果 (详细名称):\n")
            f.write("-" * 60 + "\n")
            for name, value in measurements.items():
                f.write(f"{name:30s}: {value:8.2f} cm\n")

        print(f"\n结果已保存到: {output_dir}")

    def _create_overlay_visualization(self, smpl_vertices, smpl_faces, pointcloud, output_dir):
        """创建SMPL模型与原始点云的叠加可视化"""
        try:
            import plotly.graph_objects as go

            print("生成叠加可视化...")

            # 对点云进行采样以提高性能（如果点云太大）
            if len(pointcloud) > 10000:
                indices = np.random.choice(len(pointcloud), 10000, replace=False)
                sampled_pc = pointcloud[indices]
            else:
                sampled_pc = pointcloud

            # 创建点云散点图（蓝色）
            pointcloud_trace = go.Scatter3d(
                x=sampled_pc[:, 0],
                y=sampled_pc[:, 1],
                z=sampled_pc[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color='blue',
                    opacity=0.6
                ),
                name='原始点云',
                showlegend=True
            )

            # 创建SMPL网格（半透明红色）
            smpl_mesh = go.Mesh3d(
                x=smpl_vertices[:, 0],
                y=smpl_vertices[:, 1],
                z=smpl_vertices[:, 2],
                i=smpl_faces[:, 0],
                j=smpl_faces[:, 1],
                k=smpl_faces[:, 2],
                color='lightcoral',
                opacity=0.5,
                name='拟合的SMPL模型',
                showscale=False,
                flatshading=False,
                lighting=dict(
                    diffuse=0.5,
                    specular=0.3,
                    roughness=0.7
                ),
                lightposition=dict(x=0, y=0, z=-1)
            )

            # 创建图形
            fig = go.Figure(data=[pointcloud_trace, smpl_mesh])

            # 设置布局
            fig.update_layout(
                title='SMPL模型拟合结果 - 叠加可视化',
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z',
                    aspectmode='data',
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    )
                ),
                showlegend=True,
                width=1200,
                height=900
            )

            # 保存HTML文件
            html_path = os.path.join(output_dir, "overlay_visualization.html")
            fig.write_html(html_path)
            print(f"已保存叠加可视化: {html_path}")
            print(f"  可在浏览器中打开查看: file://{os.path.abspath(html_path)}")

        except ImportError:
            print("警告: 未安装plotly，跳过可视化生成。安装命令: pip install plotly")
        except Exception as e:
            print(f"警告: 生成可视化时出错: {e}")


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
    parser.add_argument('--gender', type=str, default='male',
                        choices=['male', 'female', 'neutral'],
                        help='性别')
    parser.add_argument('--model_path', type=str, default='data',
                        help='SMPL模型路径')
    parser.add_argument('--keypoint_iterations', type=int, default=300,
                        help='关键点拟合迭代次数')
    parser.add_argument('--pointcloud_iterations', type=int, default=200,
                        help='点云拟合迭代次数')
    parser.add_argument('--freeze_pose', action='store_true', default=True,
                        help='冻结姿态（只优化body shape），适用于不完整点云')
    parser.add_argument('--optimize_pose', action='store_true',
                        help='优化姿态（不推荐用于不完整点云）')
    parser.add_argument('--pose_type', type=str, default='apose',
                        choices=['apose', 'tpose', 'natural', 'zero'],
                        help='初始姿态类型 (apose=手臂向外30度, tpose=手臂水平, natural=手臂下垂, zero=标准零姿态)')
    parser.add_argument('--coordinate_transform', type=str, default='flip_y',
                        choices=['flip_y', 'flip_z', 'swap_yz', 'swap_yz_flip', 'none'],
                        help='坐标系转换 (flip_y=翻转Y轴, flip_z=翻转Z轴, swap_yz=交换YZ, swap_yz_flip=交换YZ并翻转Y, none=不转换)')
    parser.add_argument('--visualize', action='store_true',
                        help='是否可视化结果')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='计算设备 (auto=自动选择, cpu=强制CPU, cuda=强制GPU)')
    parser.add_argument('--num_samples', type=int, default=2000,
                        help='点云拟合时的采样点数 (默认2000，设为-1使用全部点)')
    parser.add_argument('--betas_reg', type=float, default=0.001,
                        help='形状正则化权重 (默认0.001，值越小允许形状变化越大)')
    parser.add_argument('--pose_reg', type=float, default=0.0001,
                        help='姿态正则化权重 (默认0.0001)')
    parser.add_argument('--bidirectional_weight', type=float, default=0.5,
                        help='双向Chamfer距离中模型→点云的权重 (默认0.5，设为0则只用单向距离)')

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

    # 应用坐标系转换（仅对点云）
    if pointcloud is not None and args.coordinate_transform != 'none':
        print("\n应用坐标系转换...")
        pointcloud = fitter.transform_coordinate_system(pointcloud, args.coordinate_transform)
        print(f"  转换后坐标范围:")
        print(f"    X: [{pointcloud[:, 0].min():.3f}, {pointcloud[:, 0].max():.3f}]")
        print(f"    Y: [{pointcloud[:, 1].min():.3f}, {pointcloud[:, 1].max():.3f}]")
        print(f"    Z: [{pointcloud[:, 2].min():.3f}, {pointcloud[:, 2].max():.3f}]")

    # 根据数据类型选择拟合方式
    if keypoints_3d is not None and keypoints_valid is not None:
        # NPZ格式：有关键点数据，进行关键点拟合
        print("\n开始SMPL关键点拟合...")
        betas_kp, pose_kp = fitter.fit_to_keypoints(
            keypoints_3d, keypoints_valid,
            num_iterations=args.keypoint_iterations
        )
    else:
        # PLY/OBJ格式：只有点云数据，跳过关键点拟合
        print("\n检测到点云数据（无关键点），跳过关键点拟合步骤...")
        betas_kp, pose_kp = None, None

    print("\n开始SMPL点云拟合...")
    freeze_pose = not args.optimize_pose  # 默认冻结姿态
    num_samples = args.num_samples if args.num_samples > 0 else len(pointcloud)
    betas_final, pose_final, transl_final = fitter.fit_to_pointcloud(
        pointcloud,
        initial_betas=betas_kp,
        initial_pose=pose_kp,
        num_iterations=args.pointcloud_iterations,
        num_samples=num_samples,
        freeze_pose=freeze_pose,
        betas_reg_weight=args.betas_reg,
        pose_reg_weight=args.pose_reg,
        bidirectional_weight=args.bidirectional_weight,
        pose_type=args.pose_type
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
                       measurements, labeled_measurements, transl_final, pointcloud)
    
    if args.visualize:
        fitter.visualize_results()
    
    print("\n处理完成！")


if __name__ == "__main__":
    main()
