"""
测试不同的手臂角度，找到最佳初始姿态
"""
import numpy as np
import torch
import smplx
import trimesh
import os


def create_smpl_with_arm_pose(shoulder_angle_deg, model_path="data", gender="neutral"):
    """创建指定手臂角度的SMPL模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = smplx.create(
        model_path=model_path,
        model_type='smpl',
        gender=gender,
        num_betas=10,
        use_face_contour=False,
        ext='pkl'
    ).to(device)

    # 创建pose参数
    pose = torch.zeros(72, dtype=torch.float32, device=device)

    # 设置肩关节角度
    angle_rad = np.deg2rad(shoulder_angle_deg)
    # 左肩 (body_pose index 39-41, 对应joint 16)
    pose[40] = angle_rad  # 绕Y轴旋转
    # 右肩 (body_pose index 42-44, 对应joint 17)
    pose[43] = -angle_rad

    betas = torch.zeros(10, dtype=torch.float32, device=device)

    with torch.no_grad():
        output = model(
            betas=betas.unsqueeze(0),
            body_pose=pose[3:].unsqueeze(0),
            global_orient=pose[:3].unsqueeze(0),
            return_verts=True
        )
        vertices = output.vertices[0].cpu().numpy()

    return vertices, model.faces


def main():
    output_dir = "./smpl_arm_poses"
    os.makedirs(output_dir, exist_ok=True)

    # 测试不同角度
    angles = [0, 15, 30, 45, 60, 75, 90]  # 0=T-pose, 90=完全下垂

    print("生成不同手臂角度的SMPL模型...")
    for angle in angles:
        print(f"\n角度: {angle}°")
        vertices, faces = create_smpl_with_arm_pose(angle)

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.visual.vertex_colors = [255, 0, 0, 150]

        output_path = os.path.join(output_dir, f"smpl_arms_{angle}deg.ply")
        mesh.export(output_path)
        print(f"  已保存: {output_path}")

    print(f"\n所有模型已保存到: {output_dir}")
    print("请在CloudCompare中对比不同角度，找到最接近点云的姿态")
    print("然后告诉我角度值，我会更新代码")


if __name__ == "__main__":
    main()
