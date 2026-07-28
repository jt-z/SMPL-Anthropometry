import numpy as np
import trimesh
import smplx
import torch
import argparse
import os


def visualize_fitting_result(pointcloud_path, smpl_params_path, model_path="data", gender="neutral"):
    """可视化点云和拟合的SMPL模型"""

    # 加载点云
    print(f"加载点云: {pointcloud_path}")
    mesh = trimesh.load(pointcloud_path)
    pointcloud = np.array(mesh.vertices)
    print(f"  点云点数: {len(pointcloud)}")

    # 加载SMPL参数
    print(f"加载SMPL参数: {smpl_params_path}")
    params = np.load(smpl_params_path)
    betas = params['betas']
    pose = params['pose']
    transl = params['transl'] if 'transl' in params else np.zeros(3)
    print(f"  平移参数: {transl}")

    # 创建SMPL模型
    print("创建SMPL模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = smplx.create(
        model_path=model_path,
        model_type='smpl',
        gender=gender,
        num_betas=10,
        use_face_contour=False,
        ext='pkl'
    ).to(device)

    # 生成SMPL网格
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
        joints = output.joints[0].cpu().numpy()

    # 应用平移
    vertices = vertices + transl
    joints = joints + transl

    # 创建SMPL mesh
    faces = model.faces
    smpl_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # 可视化
    print("创建可视化...")

    # 点云 - 蓝色
    pc_cloud = trimesh.PointCloud(pointcloud)
    pc_cloud.colors = np.tile([0, 0, 255, 255], (len(pointcloud), 1))

    # SMPL mesh - 红色半透明
    smpl_mesh.visual.vertex_colors = np.tile([255, 0, 0, 150], (len(vertices), 1))

    # 保存文件
    output_dir = os.path.dirname(smpl_params_path)

    # 1. 保存点云（蓝色）
    pointcloud_path = os.path.join(output_dir, "pointcloud_blue.ply")
    pc_cloud.export(pointcloud_path)
    print(f"点云已保存到: {pointcloud_path}")

    # 2. 保存SMPL网格（红色）
    smpl_mesh_path = os.path.join(output_dir, "smpl_fitted_red.ply")
    smpl_mesh.export(smpl_mesh_path)
    print(f"SMPL网格已保存到: {smpl_mesh_path}")

    # 3. 保存合并场景
    scene = trimesh.Scene([pc_cloud, smpl_mesh])
    combined_path = os.path.join(output_dir, "combined_blue_red.ply")
    try:
        # 导出为单个PLY（可能会丢失颜色）
        combined_mesh = trimesh.util.concatenate([
            trimesh.PointCloud(pointcloud, colors=[0, 0, 255, 255]),
            smpl_mesh
        ])
        combined_mesh.export(combined_path)
        print(f"合并场景已保存到: {combined_path}")
    except Exception as e:
        print(f"保存合并场景失败: {e}")
        # 尝试导出为GLB格式保留颜色
        glb_path = os.path.join(output_dir, "combined_scene.glb")
        scene.export(glb_path)
        print(f"合并场景已保存为GLB: {glb_path}")

    print(f"\n拟合结果统计:")
    print(f"  点云中心: {pointcloud.mean(axis=0)}")
    print(f"  SMPL中心: {vertices.mean(axis=0)}")
    print(f"  点云范围: X[{pointcloud[:, 0].min():.3f}, {pointcloud[:, 0].max():.3f}] "
          f"Y[{pointcloud[:, 1].min():.3f}, {pointcloud[:, 1].max():.3f}] "
          f"Z[{pointcloud[:, 2].min():.3f}, {pointcloud[:, 2].max():.3f}]")
    print(f"  SMPL范围: X[{vertices[:, 0].min():.3f}, {vertices[:, 0].max():.3f}] "
          f"Y[{vertices[:, 1].min():.3f}, {vertices[:, 1].max():.3f}] "
          f"Z[{vertices[:, 2].min():.3f}, {vertices[:, 2].max():.3f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='可视化点云拟合结果')
    parser.add_argument('--pointcloud', type=str, required=True, help='点云文件路径')
    parser.add_argument('--params', type=str, required=True, help='SMPL参数文件路径')
    parser.add_argument('--model_path', type=str, default='data', help='SMPL模型路径')
    parser.add_argument('--gender', type=str, default='neutral', help='性别')

    args = parser.parse_args()

    visualize_fitting_result(args.pointcloud, args.params, args.model_path, args.gender)
