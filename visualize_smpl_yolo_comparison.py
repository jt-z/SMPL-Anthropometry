#!/usr/bin/env python3
"""
可视化SMPL模型与YOLO关键点对比
同时展示：
1. SMPL模型的网格和关节
2. YOLO检测到的3D关键点
3. 骨架连线
"""

import numpy as np
import torch
import smplx
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def get_device():
    if torch.cuda.is_available():
        try:
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
            return torch.device('cuda')
        except:
            return torch.device('cpu')
    return torch.device('cpu')

def load_smpl_model(betas, gender='neutral'):
    """加载SMPL模型"""
    device = get_device()
    
    model = smplx.create(
        model_path="data",
        model_type="smpl",
        gender=gender,
        num_betas=10,
        use_face_contour=False,
        ext='pkl'
    ).to(device)
    
    betas_torch = torch.tensor(betas, dtype=torch.float32).unsqueeze(0).to(device)
    pose_torch = torch.zeros(1, 72).to(device)
    
    output = model(
        betas=betas_torch,
        body_pose=pose_torch[:, 3:],
        global_orient=pose_torch[:, :3],
        transl=torch.zeros(1, 3, device=device),
        return_verts=True,
        return_joints=True
    )
    
    vertices = output.vertices.detach().cpu().numpy()[0]
    joints = output.joints.detach().cpu().numpy()[0]
    
    return vertices, joints, model.faces

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

def transform_yolo_to_smpl(yolo_kps, scale=1.964, flip_y=True):
    """将YOLO坐标系转换到SMPL坐标系"""
    result = {}
    
    left_hip = yolo_kps.get('left_hip')
    right_hip = yolo_kps.get('right_hip')
    
    if left_hip is not None and right_hip is not None:
        hip_center = (left_hip + right_hip) / 2
    else:
        hip_center = np.mean([v for v in yolo_kps.values()], axis=0)
    
    for name, coords in yolo_kps.items():
        centered = coords - hip_center
        if flip_y:
            centered[1] = -centered[1]
        result[name] = centered / 1000.0 * scale
    
    return result

def create_smpl_mesh(vertices, faces):
    """创建SMPL网格"""
    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color='lightpink',
        opacity=0.3,
        name='SMPL Model',
        showscale=False,
        flatshading=False
    )

def create_keypoint_markers(keypoints, name_prefix='', color='blue', size=6):
    """创建关键点标记"""
    names = list(keypoints.keys())
    coords = np.array(list(keypoints.values()))
    
    return go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode='markers+text',
        marker=dict(size=size, color=color, opacity=1.0),
        text=names,
        textposition='top center',
        textfont=dict(size=8),
        name=name_prefix,
        showlegend=True
    )

def create_skeleton_lines(keypoints, coco_order, color='green', name=''):
    """创建骨架连线"""
    segments = [
        ('nose', 'left_shoulder'),
        ('nose', 'right_shoulder'),
        ('left_shoulder', 'right_shoulder'),
        ('left_shoulder', 'left_elbow'),
        ('left_elbow', 'left_wrist'),
        ('right_shoulder', 'right_elbow'),
        ('right_elbow', 'right_wrist'),
        ('left_shoulder', 'left_hip'),
        ('right_shoulder', 'right_hip'),
        ('left_hip', 'right_hip'),
        ('left_hip', 'left_knee'),
        ('left_knee', 'left_ankle'),
        ('right_hip', 'right_knee'),
        ('right_knee', 'right_ankle'),
    ]
    
    x_lines = []
    y_lines = []
    z_lines = []
    
    for start, end in segments:
        if start in keypoints and end in keypoints:
            x_lines.extend([keypoints[start][0], keypoints[end][0], None])
            y_lines.extend([keypoints[start][1], keypoints[end][1], None])
            z_lines.extend([keypoints[start][2], keypoints[end][2], None])
    
    return go.Scatter3d(
        x=x_lines,
        y=y_lines,
        z=z_lines,
        mode='lines',
        line=dict(color=color, width=3),
        name=name,
        showlegend=True
    )

def main():
    betas = np.array([1.07, 2.94, -4.15, -0.93, -1.99, -4.73, 0.77, -2.42, -3.43, -2.31])
    
    txt_path = "/home/zjt/dev/On_Git_Projects/3D-Human-Measure/frame_output/frame_1860_yolo_measure_results.txt"
    
    print("Loading SMPL model...")
    vertices, joints, faces = load_smpl_model(betas, 'neutral')
    
    print(f"Loading YOLO keypoints from {txt_path}...")
    yolo_kps = parse_yolo_keypoints(txt_path)
    
    smpl_kps = {}
    smpl_joint_names = [
        'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 
        'right_knee', 'spine2', 'left_ankle', 'right_ankle', 'spine3', 
        'left_foot', 'right_foot', 'neck', 'left_shoulder', 'right_shoulder', 
        'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hand', 
        'right_hand', 'jaw', 'left_eye', 'right_eye'
    ]
    
    for i, name in enumerate(smpl_joint_names):
        if i < len(joints):
            smpl_kps[name] = joints[i]
    
    yolo_transformed = transform_yolo_to_smpl(yolo_kps, scale=1.0, flip_y=True)
    
    fig = go.Figure()
    
    fig.add_trace(create_smpl_mesh(vertices, faces))
    
    for name in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip',
                 'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
                 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
                 'neck']:
        if name in smpl_kps:
            fig.add_trace(go.Scatter3d(
                x=[smpl_kps[name][0]],
                y=[smpl_kps[name][1]],
                z=[smpl_kps[name][2]],
                mode='markers',
                marker=dict(size=8, color='red', opacity=1.0),
                name=f'SMPL {name}',
                showlegend=True
            ))
    
    fig.add_trace(create_keypoint_markers(yolo_transformed, 'YOLO ', 'blue', 6))
    fig.add_trace(create_skeleton_lines(yolo_transformed, None, 'blue', 'YOLO Skeleton'))
    
    fig.update_layout(
        title=dict(
            text="SMPL Model + YOLO Keypoints Comparison<br><sub>Blue: YOLO, Red: SMPL Joints</sub>",
            x=0.5
        ),
        scene=dict(
            aspectmode='data',
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=1400,
        height=900,
        showlegend=True,
        legend=dict(x=0.02, y=0.98)
    )
    
    output_path = "smpl_yolo_comparison.html"
    fig.write_html(output_path)
    print(f"\nComparison visualization saved to: {output_path}")
    print(f"Open this file in a web browser to view.")

if __name__ == "__main__":
    main()
