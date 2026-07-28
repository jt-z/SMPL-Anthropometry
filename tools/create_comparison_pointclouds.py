"""
创建用于对比的点云文件：
- 原始点云（蓝色）
- SMPL采样点云（红色，稀疏化）
"""
import numpy as np
import trimesh
import smplx
import torch
import argparse


def sample_mesh_points(mesh, num_samples=5000):
    """从网格表面均匀采样点"""
    points, face_indices = trimesh.sample.sample_surface(mesh, num_samples)
    return points


def main():
    parser = argparse.ArgumentParser(description='创建对比点云')
    parser.add_argument('--pointcloud', type=str, required=True, help='原始点云路径')
    parser.add_argument('--params', type=str, required=True, help='SMPL参数路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--smpl_samples', type=int, default=5000, help='SMPL采样点数')
    parser.add_argument('--model_path', type=str, default='data', help='SMPL模型路径')
    parser.add_argument('--gender', type=str, default='neutral', help='性别')

    args = parser.parse_args()

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载原始点云
    print("加载原始点云...")
    mesh_or_pc = trimesh.load(args.pointcloud)
    if hasattr(mesh_or_pc, 'vertices'):
        original_pc = np.array(mesh_or_pc.vertices)
    else:
        original_pc = np.array(mesh_or_pc.vertices)
    print(f"  点数: {len(original_pc)}")
    print(f"  中心: {original_pc.mean(axis=0)}")

    # 2. 生成SMPL网格
    print("生成SMPL网格...")
    params = np.load(args.params)
    betas = params['betas']
    pose = params['pose']
    transl = params['transl'] if 'transl' in params else np.zeros(3)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = smplx.create(
        model_path=args.model_path,
        model_type='smpl',
        gender=args.gender,
        num_betas=10,
        use_face_contour=False,
        ext='pkl'
    ).to(device)

    betas_torch = torch.tensor(betas, dtype=torch.float32).unsqueeze(0).to(device)
    pose_torch = torch.tensor(pose, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(
            betas=betas_torch,
            body_pose=pose_torch[:, 3:],
            global_orient=pose_torch[:, :3],
            return_verts=True
        )
        vertices = output.vertices[0].cpu().numpy()

    vertices = vertices + transl
    faces = model.faces
    smpl_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    print(f"  顶点数: {len(vertices)}")
    print(f"  中心: {vertices.mean(axis=0)}")

    # 3. 从SMPL网格采样点
    print(f"从SMPL采样 {args.smpl_samples} 个点...")
    smpl_points = sample_mesh_points(smpl_mesh, args.smpl_samples)
    print(f"  采样点数: {len(smpl_points)}")

    # 4. 保存蓝色点云（原始）
    blue_cloud = trimesh.PointCloud(original_pc)
    blue_colors = np.tile([0, 0, 255, 255], (len(original_pc), 1))
    blue_cloud.colors = blue_colors

    blue_path = os.path.join(args.output_dir, "original_blue.ply")
    blue_cloud.export(blue_path)
    print(f"\n蓝色点云已保存: {blue_path}")

    # 5. 保存红色点云（SMPL采样）
    red_cloud = trimesh.PointCloud(smpl_points)
    red_colors = np.tile([255, 0, 0, 255], (len(smpl_points), 1))
    red_cloud.colors = red_colors

    red_path = os.path.join(args.output_dir, "smpl_red.ply")
    red_cloud.export(red_path)
    print(f"红色点云已保存: {red_path}")

    # 6. 合并保存
    all_points = np.vstack([original_pc, smpl_points])
    all_colors = np.vstack([blue_colors, red_colors])

    combined_cloud = trimesh.PointCloud(all_points)
    combined_cloud.colors = all_colors

    combined_path = os.path.join(args.output_dir, "comparison.ply")
    combined_cloud.export(combined_path)
    print(f"合并点云已保存: {combined_path}")

    print("\n统计信息:")
    print(f"  原始点云: {len(original_pc)} 点 (蓝色)")
    print(f"  SMPL采样: {len(smpl_points)} 点 (红色)")
    print(f"  总计: {len(all_points)} 点")
    print(f"\n中心距离: {np.linalg.norm(original_pc.mean(axis=0) - smpl_points.mean(axis=0)):.4f} 米")


if __name__ == "__main__":
    main()
