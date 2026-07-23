#!/usr/bin/env python3
"""
可视化SMPL模型与YOLO关键点对比
"""

import numpy as np
import torch
import smplx
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
        opacity=0.5,
        name='SMPL Model',
        showscale=False,
        flatshading=False,
        lighting=dict(
            diffuse=0.5,
            specular=0.3,
            roughness=0.7
        ),
        lightposition=dict(x=0, y=0, z=-1)
    )

def create_joint_sphere(position, name, color='blue', size=5):
    """创建关节球"""
    return go.Scatter3d(
        x=[position[0]],
        y=[position[1]],
        z=[position[2]],
        mode='markers',
        marker=dict(size=size, color=color, opacity=1.0),
        name=name,
        showlegend=True
    )

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualize SMPL Model")
    parser.add_argument('--betas', type=str, default=None,
                        help="Path to betas.npz or comma-separated betas values")
    parser.add_argument('--gender', type=str, default='neutral',
                        choices=['male', 'female', 'neutral'])
    parser.add_argument('--input', type=str,
                        default="/home/zjt/dev/On_Git_Projects/3D-Human-Measure/frame_output/frame_1860_yolo_measure_results.txt",
                        help="Input TXT file for YOLO keypoints")
    args = parser.parse_args()
    
    if args.betas is None:
        betas = np.array([1.07, 2.94, -4.15, -0.93, -1.99, -4.73, 0.77, -2.42, -3.43, -2.31])
    elif args.betas.endswith('.npz'):
        data = np.load(args.betas)
        betas = data['betas']
    else:
        betas = np.array([float(x) for x in args.betas.split(',')])
    
    print(f"Loading SMPL model with betas: {betas}")
    
    vertices, joints, faces = load_smpl_model(betas, args.gender)
    
    print(f"Vertices shape: {vertices.shape}")
    print(f"Joints shape: {joints.shape}")
    
    smpl_joint_names = [
        'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 
        'right_knee', 'spine2', 'left_ankle', 'right_ankle', 'spine3', 
        'left_foot', 'right_foot', 'neck', 'left_shoulder', 'right_shoulder', 
        'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hand', 
        'right_hand', 'jaw', 'left_eye', 'right_eye'
    ]
    
    fig = go.Figure()
    
    fig.add_trace(create_smpl_mesh(vertices, faces))
    
    for i, name in enumerate(smpl_joint_names):
        if i < len(joints):
            fig.add_trace(create_joint_sphere(joints[i], name, color='red', size=4))
    
    fig.update_layout(
        title=dict(
            text=f"SMPL Model Visualization<br><sub>Gender: {args.gender}, Height: {joints[0,1]*100:.1f}cm</sub>",
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
        width=1200,
        height=800,
        showlegend=True,
        legend=dict(x=0.02, y=0.98)
    )
    
    output_path = "smpl_visualization.html"
    fig.write_html(output_path)
    print(f"\nVisualization saved to: {output_path}")
    print(f"You can open this file in a web browser to view the 3D model.")
    
    fig.show()

if __name__ == "__main__":
    main()
