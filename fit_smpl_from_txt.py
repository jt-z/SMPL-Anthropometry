import numpy as np
import torch
import smplx
import os
import re
from measure import MeasureBody
from measurement_definitions import STANDARD_LABELS
import argparse
import warnings


def get_device():
    if torch.cuda.is_available():
        try:
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
            return torch.device('cuda')
        except Exception as e:
            warnings.warn(f"CUDA available but initialization failed: {e}")
            warnings.warn("Will use CPU")
            return torch.device('cpu')
    else:
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
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        keypoints_3d = []
        confidences = []
        
        for kp_name in self.coco_keypoint_names:
            pattern = rf'{kp_name}\s+:.+?3D=\[\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\]'
            match = re.search(pattern, content)
            
            if match:
                x = float(match.group(1))
                y = float(match.group(2))
                z = float(match.group(3))
                keypoints_3d.append([x, y, z])
                confidences.append(0.9)
            else:
                keypoints_3d.append([0, 0, 0])
                confidences.append(0.0)
        
        keypoints_3d = np.array(keypoints_3d, dtype=np.float32)
        confidences = np.array(confidences, dtype=np.float32)
        
        keypoints_with_conf = np.concatenate([keypoints_3d, confidences.reshape(-1, 1)], axis=1)
        
        keypoints_valid = confidences > 0
        
        pointcloud = self.generate_pointcloud_from_keypoints(keypoints_3d)
        
        return keypoints_with_conf, keypoints_valid, pointcloud
    
    def generate_pointcloud_from_keypoints(self, keypoints_3d):
        valid_mask = np.abs(keypoints_3d).sum(axis=1) > 0
        valid_keypoints = keypoints_3d[valid_mask]
        
        pointcloud = []
        for i, kp in enumerate(valid_keypoints):
            x, y, z = kp
            pointcloud.append([x, y, z])
            
            if i < len(valid_keypoints) - 1:
                next_kp = valid_keypoints[i + 1]
                num_samples = 20
                for t in np.linspace(0, 1, num_samples):
                    interpolated = kp + t * (next_kp - kp)
                    pointcloud.append(interpolated)
        
        skeleton_connections = [
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
            (12, 14), (14, 16)
        ]
        
        for start_idx, end_idx in skeleton_connections:
            if valid_mask[start_idx] and valid_mask[end_idx]:
                start = keypoints_3d[start_idx]
                end = keypoints_3d[end_idx]
                num_samples = 50
                for t in np.linspace(0, 1, num_samples):
                    interpolated = start + t * (end - start)
                    pointcloud.append(interpolated)
        
        return np.array(pointcloud, dtype=np.float32)
    
    def save_as_npz(self, output_path, keypoints_3d, keypoints_valid, pointcloud):
        np.savez(output_path,
                keypoints_3d=keypoints_3d,
                keypoints_valid=keypoints_valid,
                pointcloud=pointcloud)
        print(f"数据已保存到: {output_path}")


class SMPLFitterFromMeasurements:
    def __init__(self, model_path="data", model_type="smpl", gender="neutral", device=None):
        self.model_type = model_type
        self.model_path = model_path
        self.gender = gender
        
        if device is None:
            self.device = get_device()
        else:
            self.device = device
        
        if self.device.type == 'cuda':
            print(f"Device: CUDA (GPU)")
        else:
            print(f"Device: CPU")
        
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
            0: 15,
            5: 16,
            6: 17,
            7: 18,
            8: 19,
            9: 20,
            10: 21,
            11: 1,
            12: 2,
            13: 4,
            14: 5,
            15: 7,
            16: 8,
        }
        
        self.smpl_joint_names = {
            0: 'pelvis', 1: 'left_hip', 2: 'right_hip', 3: 'spine1',
            4: 'left_knee', 5: 'right_knee', 6: 'spine2',
            7: 'left_ankle', 8: 'right_ankle', 9: 'spine3',
            10: 'left_foot', 11: 'right_foot', 12: 'neck',
            13: 'left_collar', 14: 'right_collar', 15: 'head',
            16: 'left_shoulder', 17: 'right_shoulder', 18: 'left_elbow',
            19: 'right_elbow', 20: 'left_wrist', 21: 'right_wrist',
            22: 'left_hand', 23: 'right_hand',
        }
    
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
        print("\nStarting SMPL keypoint fitting...")
        
        target_keypoints = []
        smpl_joint_indices = []
        weights = []
        
        for coco_idx, smpl_idx in self.coco_to_smpl_mapping.items():
            if keypoints_valid[coco_idx]:
                target_keypoints.append(keypoints_3d[coco_idx, :3])
                smpl_joint_indices.append(smpl_idx)
                weights.append(keypoints_3d[coco_idx, 3] if keypoints_3d.shape[1] > 3 else 0.9)
        
        if len(target_keypoints) == 0:
            print("No valid keypoints for fitting!")
            return None, None
        
        target_keypoints = np.array(target_keypoints)
        weights = np.array(weights)
        
        print(f"Using {len(target_keypoints)} keypoints for fitting")
        
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
            joints_torch = torch.tensor(joints, dtype=torch.float32, device=self.device)
            
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
                print(f"  Iteration {iteration}: Loss = {total_loss.item():.6f}")
        
        print(f"Keypoint fitting complete, best loss: {best_loss:.6f}")
        
        return best_betas.numpy(), best_pose.numpy()
    
    def fit_to_measurements(self, txt_path, initial_betas=None, num_iterations=500):
        print("\nFitting SMPL to skeleton measurements...")
        
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        measurements = {}
        
        patterns = {
            'shoulder_width': r'shoulder_width:\s*([\d.]+)\s*mm',
            'left_arm_length': r'left_arm_length:\s*([\d.]+)\s*mm',
            'right_arm_length': r'right_arm_length:\s*([\d.]+)\s*mm',
            'left_forearm_length': r'left_forearm_length:\s*([\d.]+)\s*mm',
            'right_forearm_length': r'right_forearm_length:\s*([\d.]+)\s*mm',
            'hip_width': r'hip_width:\s*([\d.]+)\s*mm',
            'left_leg_length': r'left_leg_length:\s*([\d.]+)\s*mm',
            'right_leg_length': r'right_leg_length:\s*([\d.]+)\s*mm',
            'left_shin_length': r'left_shin_length:\s*([\d.]+)\s*mm',
            'right_shin_length': r'right_shin_length:\s*([\d.]+)\s*mm',
            'torso_height': r'torso_height:\s*([\d.]+)\s*mm',
        }
        
        for name, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                measurements[name] = float(match.group(1)) / 1000.0
        
        print(f"Loaded {len(measurements)} measurements")
        for name, value in measurements.items():
            print(f"  {name}: {value*1000:.2f} mm")
        
        if initial_betas is None:
            betas = torch.zeros(10, dtype=torch.float32, requires_grad=True, device=self.device)
        else:
            betas = torch.tensor(initial_betas, dtype=torch.float32, requires_grad=True, device=self.device)
        
        optimizer = torch.optim.Adam([betas], lr=0.005)
        
        best_loss = float('inf')
        best_betas = betas.detach().clone()
        
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            
            vertices, joints = self.get_smpl_joints(betas, torch.zeros(72, device=self.device))
            
            loss = 0.0
            
            if 'shoulder_width' in measurements:
                left_shoulder = joints[16]
                right_shoulder = joints[17]
                pred_shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
                loss += (pred_shoulder_width - measurements['shoulder_width']) ** 2
            
            if 'hip_width' in measurements:
                left_hip = joints[1]
                right_hip = joints[2]
                pred_hip_width = np.linalg.norm(left_hip - right_hip)
                loss += (pred_hip_width - measurements['hip_width']) ** 2
            
            if 'left_arm_length' in measurements:
                left_shoulder = joints[16]
                left_elbow = joints[18]
                left_wrist = joints[20]
                pred_left_arm = np.linalg.norm(left_shoulder - left_elbow) + np.linalg.norm(left_elbow - left_wrist)
                loss += (pred_left_arm - measurements['left_arm_length']) ** 2
            
            if 'right_arm_length' in measurements:
                right_shoulder = joints[17]
                right_elbow = joints[19]
                right_wrist = joints[21]
                pred_right_arm = np.linalg.norm(right_shoulder - right_elbow) + np.linalg.norm(right_elbow - right_wrist)
                loss += (pred_right_arm - measurements['right_arm_length']) ** 2
            
            if 'left_leg_length' in measurements:
                left_hip = joints[1]
                left_knee = joints[4]
                left_ankle = joints[7]
                pred_left_leg = np.linalg.norm(left_hip - left_knee) + np.linalg.norm(left_knee - left_ankle)
                loss += (pred_left_leg - measurements['left_leg_length']) ** 2
            
            if 'right_leg_length' in measurements:
                right_hip = joints[2]
                right_knee = joints[5]
                right_ankle = joints[8]
                pred_right_leg = np.linalg.norm(right_hip - right_knee) + np.linalg.norm(right_knee - right_ankle)
                loss += (pred_right_leg - measurements['right_leg_length']) ** 2
            
            betas_reg = 0.001 * torch.sum(betas ** 2)
            total_loss = loss + betas_reg
            
            total_loss.backward()
            optimizer.step()
            
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_betas = betas.detach().clone()
            
            if iteration % 50 == 0:
                print(f"  Iteration {iteration}: Loss = {total_loss.item():.6f}")
        
        print(f"Measurement fitting complete, best loss: {best_loss:.6f}")
        
        return best_betas.numpy(), measurements
    
    def measure_body(self, betas):
        print("\nStarting body measurements...")
        
        betas_torch = torch.tensor(betas, dtype=torch.float32).unsqueeze(0)
        
        self.measurer.from_body_model(gender=self.gender.upper(), shape=betas_torch)
        
        measurement_names = self.measurer.all_possible_measurements
        self.measurer.measure(measurement_names)
        
        self.measurer.label_measurements(STANDARD_LABELS)
        
        measurements = self.measurer.measurements
        labeled_measurements = self.measurer.labeled_measurements
        
        print(f"Completed {len(measurements)} measurements")
        
        return measurements, labeled_measurements
    
    def visualize_results(self):
        print("\nGenerating visualization...")
        self.measurer.visualize()
    
    def save_results(self, output_dir, betas, measurements, labeled_measurements):
        os.makedirs(output_dir, exist_ok=True)
        
        np.savez(
            os.path.join(output_dir, "smpl_params.npz"),
            betas=betas
        )
        
        with open(os.path.join(output_dir, "measurements.txt"), 'w', encoding='utf-8') as f:
            f.write("SMPL 身体测量结果\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("SMPL 参数:\n")
            f.write(f"  betas: {betas}\n\n")
            
            f.write("测量结果 (标准标签):\n")
            f.write("-" * 60 + "\n")
            for label, value in labeled_measurements.items():
                f.write(f"{label:3s}: {value:8.2f} cm\n")
            
            f.write("\n测量结果 (详细名称):\n")
            f.write("-" * 60 + "\n")
            for name, value in measurements.items():
                f.write(f"{name:30s}: {value:8.2f} cm\n")
        
        print(f"\nResults saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Fit SMPL model from measurement TXT file')
    parser.add_argument('--input', type=str,
                        default='/home/zjt/dev/On_Git_Projects/3D-Human-Measure/frame_output/frame_1860_yolo_measure_results.txt',
                        help='Input TXT measurement file')
    parser.add_argument('--output', type=str,
                        default='./output_from_txt',
                        help='Output directory')
    parser.add_argument('--model_type', type=str, default='smpl',
                        choices=['smpl', 'smplx'],
                        help='Model type')
    parser.add_argument('--gender', type=str, default='neutral',
                        choices=['male', 'female', 'neutral'],
                        help='Gender')
    parser.add_argument('--model_path', type=str, default='data',
                        help='SMPL model path')
    parser.add_argument('--keypoint_iterations', type=int, default=300,
                        help='Keypoint fitting iterations')
    parser.add_argument('--measurement_iterations', type=int, default=500,
                        help='Measurement fitting iterations')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize results')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device (auto=auto select, cpu=force CPU, cuda=force GPU)')
    parser.add_argument('--save_npz', action='store_true',
                        help='Save data as npz')
    
    args = parser.parse_args()
    
    if args.device == 'auto':
        device = None
    elif args.device == 'cpu':
        device = torch.device('cpu')
    elif args.device == 'cuda':
        device = torch.device('cuda')
    
    print("=" * 60)
    print("SMPL Model Fitting from TXT Measurements")
    print("=" * 60)
    print(f"Input file: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Model type: {args.model_type}")
    print(f"Gender: {args.gender}")
    print("=" * 60)
    
    loader = TXTMeasurementLoader()
    
    keypoints_3d, keypoints_valid, pointcloud = loader.parse_txt_file(args.input)
    
    print(f"\nData loaded:")
    print(f"  Keypoints: {keypoints_3d.shape}")
    print(f"  Valid keypoints: {np.sum(keypoints_valid)}/{len(keypoints_valid)}")
    print(f"  Pointcloud: {pointcloud.shape}")
    
    if args.save_npz:
        npz_path = os.path.join(args.output, "converted_data.npz")
        loader.save_as_npz(npz_path, keypoints_3d, keypoints_valid, pointcloud)
    
    fitter = SMPLFitterFromMeasurements(
        model_path=args.model_path,
        model_type=args.model_type,
        gender=args.gender,
        device=device
    )
    
    print("\nStep 1: Keypoint fitting")
    betas_kp, pose_kp = fitter.fit_to_keypoints(
        keypoints_3d, keypoints_valid,
        num_iterations=args.keypoint_iterations
    )
    
    if betas_kp is None:
        print("Keypoint fitting failed!")
        return
    
    print("\nStep 2: Measurement-based refinement")
    betas_meas, measurements_input = fitter.fit_to_measurements(
        args.input,
        initial_betas=betas_kp,
        num_iterations=args.measurement_iterations
    )
    
    print("\nStep 3: Body measurements")
    measurements, labeled_measurements = fitter.measure_body(betas_meas)
    
    print("\n" + "=" * 60)
    print("Measurement Results (Standard Labels)")
    print("=" * 60)
    for label, value in labeled_measurements.items():
        print(f"{label:3s}: {value:8.2f} cm")
    
    print("\n" + "=" * 60)
    print("Input Measurements vs SMPL Measurements")
    print("=" * 60)
    
    measurement_map = {
        'shoulder_breadth': 'shoulder_width',
        'height': 'height',
    }
    
    for smpl_name, input_name in measurement_map.items():
        if smpl_name in measurements and input_name in measurements_input:
            print(f"{smpl_name:30s}: SMPL={measurements[smpl_name]:.2f}cm, Input={measurements_input[input_name]*100:.2f}cm")
    
    fitter.save_results(args.output, betas_meas, measurements, labeled_measurements)
    
    if args.visualize:
        fitter.visualize_results()
    
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
