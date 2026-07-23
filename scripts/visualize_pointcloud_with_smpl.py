#!/usr/bin/env python3
"""
点云和SMPL模型叠加可视化工具
"""

import numpy as np
import plotly.graph_objects as go
import trimesh
import argparse
import os

def load_pointcloud(ply_path):
    """加载点云"""
    mesh = trimesh.load(ply_path)
    return mesh.vertices

def load_smpl_result(npz_path):
    """加载SMPL拟合结果"""
    data = np.load(npz_path, allow_pickle=True)
    return data

def get_smpl_vertices(betas, model_path='data', gender='neutral'):
    """生成SMPL顶点"""
    import torch
    import smplx
    
    model = smplx.create(
        model_path=model_path,
        model_type='smpl',
        gender=gender,
        use_face_contour=False,
        num_betas=10,
        ext='pkl'
    )
    
    betas_tensor = torch.tensor(betas, dtype=torch.float32).unsqueeze(0)
    output = model(betas=betas_tensor, return_verts=True)
    
    return output.vertices[0].detach().numpy()

def create_comparison_plot(pointcloud, smpl_vertices, save_path=None):
    """创建对比可视化"""
    
    fig = go.Figure()
    
    # 1. 添加原始点云（蓝色）
    fig.add_trace(go.Scatter3d(
        x=pointcloud[:, 0],
        y=pointcloud[:, 1],
        z=pointcloud[:, 2],
        mode='markers',
        name='原始点云',
        marker=dict(
            size=2,
            color='blue',
            opacity=0.6
        )
    ))
    
    # 2. 添加SMPL模型（红色）
    fig.add_trace(go.Scatter3d(
        x=smpl_vertices[:, 0],
        y=smpl_vertices[:, 1],
        z=smpl_vertices[:, 2],
        mode='markers',
        name='SMPL拟合',
        marker=dict(
            size=2,
            color='red',
            opacity=0.8
        )
    ))
    
    # 3. 设置布局
    fig.update_layout(
        title={
            'text': '点云与SMPL拟合对比',
            'x': 0.5,
            'xanchor': 'center'
        },
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        width=1200,
        height=900,
        showlegend=True
    )
    
    if save_path:
        print(f"保存HTML到: {save_path}")
        fig.write_html(save_path)
    else:
        fig.show()
    
    return fig

def main():
    parser = argparse.ArgumentParser(description='点云和SMPL模型叠加可视化')
    parser.add_argument('--pointcloud', type=str, required=True,
                        help='输入点云文件路径 (.ply)')
    parser.add_argument('--smpl', type=str, required=True,
                        help='SMPL参数文件路径 (.npz)')
    parser.add_argument('--output', type=str, default=None,
                        help='输出HTML文件路径（可选）')
    parser.add_argument('--gender', type=str, default='neutral',
                        choices=['neutral', 'male', 'female'],
                        help='性别')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("点云和SMPL模型叠加可视化")
    print("=" * 60)
    
    # 1. 加载点云
    print(f"\n加载点云: {args.pointcloud}")
    pointcloud = load_pointcloud(args.pointcloud)
    print(f"  点云顶点数: {len(pointcloud)}")
    print(f"  X范围: [{pointcloud[:, 0].min():.3f}, {pointcloud[:, 0].max():.3f}]")
    print(f"  Y范围: [{pointcloud[:, 1].min():.3f}, {pointcloud[:, 1].max():.3f}]")
    print(f"  Z范围: [{pointcloud[:, 2].min():.3f}, {pointcloud[:, 2].max():.3f}]")
    
    # 2. 加载SMPL参数
    print(f"\n加载SMPL参数: {args.smpl}")
    smpl_data = load_smpl_result(args.smpl)
    betas = smpl_data['betas']
    print(f"  Betas: {betas}")
    
    # 3. 生成SMPL顶点
    print("\n生成SMPL模型顶点...")
    smpl_vertices = get_smpl_vertices(betas, gender=args.gender)
    print(f"  SMPL顶点数: {len(smpl_vertices)}")
    print(f"  X范围: [{smpl_vertices[:, 0].min():.3f}, {smpl_vertices[:, 0].max():.3f}]")
    print(f"  Y范围: [{smpl_vertices[:, 1].min():.3f}, {smpl_vertices[:, 1].max():.3f}]")
    print(f"  Z范围: [{smpl_vertices[:, 2].min():.3f}, {smpl_vertices[:, 2].max():.3f}]")
    
    # 4. 创建对比图
    print("\n创建对比可视化...")
    if args.output is None:
        # 自动生成输出文件名
        output_dir = os.path.dirname(args.smpl)
        args.output = os.path.join(output_dir, 'comparison_pointcloud_smpl.html')
    
    create_comparison_plot(pointcloud, smpl_vertices, args.output)
    
    print(f"\n完成！")
    print(f"HTML文件: {args.output}")
    print("\n可以在浏览器中打开查看3D对比效果")
    print("=" * 60)

if __name__ == '__main__':
    main()
