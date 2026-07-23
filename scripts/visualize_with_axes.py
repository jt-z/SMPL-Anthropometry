#!/usr/bin/env python3
"""
带坐标系的点云和SMPL可视化工具
"""

import numpy as np
import plotly.graph_objects as go
import trimesh
import argparse
import os

def create_coordinate_axes(origin=[0, 0, 0], length=0.3, name_prefix=''):
    """创建坐标轴"""
    traces = []
    
    # X轴 - 红色
    traces.append(go.Scatter3d(
        x=[origin[0], origin[0] + length],
        y=[origin[1], origin[1]],
        z=[origin[2], origin[2]],
        mode='lines+text',
        name=f'{name_prefix}X轴',
        line=dict(color='red', width=5),
        text=['', 'X'],
        textposition='top center',
        textfont=dict(size=16, color='red')
    ))
    
    # Y轴 - 绿色
    traces.append(go.Scatter3d(
        x=[origin[0], origin[0]],
        y=[origin[1], origin[1] + length],
        z=[origin[2], origin[2]],
        mode='lines+text',
        name=f'{name_prefix}Y轴',
        line=dict(color='green', width=5),
        text=['', 'Y'],
        textposition='top center',
        textfont=dict(size=16, color='green')
    ))
    
    # Z轴 - 蓝色
    traces.append(go.Scatter3d(
        x=[origin[0], origin[0]],
        y=[origin[1], origin[1]],
        z=[origin[2], origin[2] + length],
        mode='lines+text',
        name=f'{name_prefix}Z轴',
        line=dict(color='blue', width=5),
        text=['', 'Z'],
        textposition='top center',
        textfont=dict(size=16, color='blue')
    ))
    
    return traces

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

def create_comparison_with_axes(pointcloud, smpl_vertices, save_path=None):
    """创建带坐标系的对比可视化"""
    
    fig = go.Figure()
    
    # 1. 计算中心点
    pc_center = pointcloud.mean(axis=0)
    smpl_center = smpl_vertices.mean(axis=0)
    
    # 2. 添加点云坐标系（在点云中心）
    pc_axes = create_coordinate_axes(pc_center, length=0.2, name_prefix='点云-')
    for trace in pc_axes:
        fig.add_trace(trace)
    
    # 3. 添加SMPL坐标系（在SMPL中心）
    smpl_axes = create_coordinate_axes(smpl_center, length=0.2, name_prefix='SMPL-')
    for trace in smpl_axes:
        fig.add_trace(trace)
    
    # 4. 添加世界坐标系（原点）
    world_axes = create_coordinate_axes([0, 0, 0], length=0.3, name_prefix='世界-')
    for trace in world_axes:
        fig.add_trace(trace)
    
    # 5. 添加原始点云（半透明蓝色）
    fig.add_trace(go.Scatter3d(
        x=pointcloud[:, 0],
        y=pointcloud[:, 1],
        z=pointcloud[:, 2],
        mode='markers',
        name='原始点云',
        marker=dict(
            size=1.5,
            color='blue',
            opacity=0.4
        )
    ))
    
    # 6. 添加SMPL模型（半透明红色）
    fig.add_trace(go.Scatter3d(
        x=smpl_vertices[:, 0],
        y=smpl_vertices[:, 1],
        z=smpl_vertices[:, 2],
        mode='markers',
        name='SMPL拟合',
        marker=dict(
            size=1.5,
            color='red',
            opacity=0.6
        )
    ))
    
    # 7. 设置布局
    fig.update_layout(
        title={
            'text': '点云与SMPL对比 + 坐标系',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        scene=dict(
            xaxis_title='X (红色)',
            yaxis_title='Y (绿色)',
            zaxis_title='Z (蓝色)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=1400,
        height=1000,
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)'
        )
    )
    
    # 8. 添加注释
    annotations_text = [
        f"点云中心: ({pc_center[0]:.3f}, {pc_center[1]:.3f}, {pc_center[2]:.3f})",
        f"SMPL中心: ({smpl_center[0]:.3f}, {smpl_center[1]:.3f}, {smpl_center[2]:.3f})",
        f"点云范围: X[{pointcloud[:, 0].min():.3f}, {pointcloud[:, 0].max():.3f}]",
        f"          Y[{pointcloud[:, 1].min():.3f}, {pointcloud[:, 1].max():.3f}]",
        f"          Z[{pointcloud[:, 2].min():.3f}, {pointcloud[:, 2].max():.3f}]",
        f"SMPL范围: X[{smpl_vertices[:, 0].min():.3f}, {smpl_vertices[:, 0].max():.3f}]",
        f"          Y[{smpl_vertices[:, 1].min():.3f}, {smpl_vertices[:, 1].max():.3f}]",
        f"          Z[{smpl_vertices[:, 2].min():.3f}, {smpl_vertices[:, 2].max():.3f}]",
    ]
    
    fig.add_annotation(
        text="<br>".join(annotations_text),
        xref="paper", yref="paper",
        x=0.02, y=0.02,
        showarrow=False,
        bgcolor="rgba(255, 255, 255, 0.9)",
        bordercolor="black",
        borderwidth=1,
        font=dict(family="monospace", size=10),
        align="left"
    )
    
    if save_path:
        print(f"保存HTML到: {save_path}")
        fig.write_html(save_path)
    else:
        fig.show()
    
    return fig

def main():
    parser = argparse.ArgumentParser(description='带坐标系的点云和SMPL模型可视化')
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
    
    print("=" * 70)
    print("点云和SMPL模型带坐标系可视化")
    print("=" * 70)
    
    # 1. 加载点云
    print(f"\n加载点云: {args.pointcloud}")
    pointcloud = load_pointcloud(args.pointcloud)
    pc_center = pointcloud.mean(axis=0)
    print(f"  点云顶点数: {len(pointcloud)}")
    print(f"  点云中心: ({pc_center[0]:.3f}, {pc_center[1]:.3f}, {pc_center[2]:.3f})")
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
    smpl_center = smpl_vertices.mean(axis=0)
    print(f"  SMPL顶点数: {len(smpl_vertices)}")
    print(f"  SMPL中心: ({smpl_center[0]:.3f}, {smpl_center[1]:.3f}, {smpl_center[2]:.3f})")
    print(f"  X范围: [{smpl_vertices[:, 0].min():.3f}, {smpl_vertices[:, 0].max():.3f}]")
    print(f"  Y范围: [{smpl_vertices[:, 1].min():.3f}, {smpl_vertices[:, 1].max():.3f}]")
    print(f"  Z范围: [{smpl_vertices[:, 2].min():.3f}, {smpl_vertices[:, 2].max():.3f}]")
    
    # 4. 分析问题
    print("\n" + "=" * 70)
    print("坐标系分析:")
    print("=" * 70)
    
    # 判断哪个轴是向上的
    pc_y_range = pointcloud[:, 1].max() - pointcloud[:, 1].min()
    pc_z_range = pointcloud[:, 2].max() - pointcloud[:, 2].min()
    smpl_y_range = smpl_vertices[:, 1].max() - smpl_vertices[:, 1].min()
    smpl_z_range = smpl_vertices[:, 2].max() - smpl_vertices[:, 2].min()
    
    print(f"点云: Y轴范围={pc_y_range:.3f}, Z轴范围={pc_z_range:.3f}")
    if pc_z_range > pc_y_range * 0.5:
        print("  → 点云可能使用Z轴向上")
    
    print(f"SMPL: Y轴范围={smpl_y_range:.3f}, Z轴范围={smpl_z_range:.3f}")
    if smpl_y_range > smpl_z_range * 2:
        print("  → SMPL使用Y轴向上（标准）")
    
    print(f"\n中心距离: {np.linalg.norm(pc_center - smpl_center):.3f}")
    print("  → 两者位置相距很远，需要对齐！")
    
    # 5. 创建带坐标系的对比图
    print("\n创建带坐标系的对比可视化...")
    if args.output is None:
        output_dir = os.path.dirname(args.smpl)
        args.output = os.path.join(output_dir, 'comparison_with_axes.html')
    
    create_comparison_with_axes(pointcloud, smpl_vertices, args.output)
    
    print(f"\n完成！")
    print(f"HTML文件: {args.output}")
    print("\n可以在浏览器中打开查看:")
    print("- 红色轴 = X轴")
    print("- 绿色轴 = Y轴")  
    print("- 蓝色轴 = Z轴")
    print("=" * 70)

if __name__ == '__main__':
    main()
