import numpy as np
import torch
from fit_smpl_from_data import SMPLFitterFromData


def quick_fit_example():
    npz_path = '/home/zjt/dev/On_Git_Projects/3D-Human-Measure/demo3_back_test/output_v2_yolov8-seg(segmenters_yolov8n-seg.pt)/smpl_input.npz'
    
    fitter = SMPLFitterFromData(
        model_path='data',
        model_type='smpl',
        gender='neutral'
    )
    
    keypoints_3d, keypoints_valid, pointcloud = fitter.load_data(npz_path)
    
    print("\n步骤1: 关键点拟合")
    betas, pose = fitter.fit_to_keypoints(
        keypoints_3d, keypoints_valid,
        num_iterations=200
    )
    
    print("\n步骤2: 身体测量")
    measurements, labeled_measurements = fitter.measure_body(betas)
    
    print("\n测量结果:")
    for label, value in labeled_measurements.items():
        print(f"{label:3s}: {value:8.2f} cm")
    
    return measurements, labeled_measurements


def detailed_fit_example():
    npz_path = '/home/zjt/dev/On_Git_Projects/3D-Human-Measure/demo3_back_test/output_v2_yolov8-seg(segmenters_yolov8n-seg.pt)/smpl_input.npz'
    
    fitter = SMPLFitterFromData(
        model_path='data',
        model_type='smpl',
        gender='neutral'
    )
    
    keypoints_3d, keypoints_valid, pointcloud = fitter.load_data(npz_path)
    
    print("\n步骤1: 关键点拟合")
    betas_kp, pose_kp = fitter.fit_to_keypoints(
        keypoints_3d, keypoints_valid,
        num_iterations=300
    )
    
    print("\n步骤2: 点云精细拟合")
    betas_final, pose_final = fitter.fit_to_pointcloud(
        pointcloud,
        initial_betas=betas_kp,
        initial_pose=pose_kp,
        num_iterations=200
    )
    
    print("\n步骤3: 身体测量")
    measurements, labeled_measurements = fitter.measure_body(betas_final)
    
    print("\n步骤4: 可视化")
    fitter.visualize_results()
    
    print("\n步骤5: 保存结果")
    fitter.save_results(
        output_dir='./output_detailed_fit',
        betas=betas_final,
        pose=pose_final,
        measurements=measurements,
        labeled_measurements=labeled_measurements
    )
    
    return measurements, labeled_measurements


if __name__ == "__main__":
    print("选择运行模式:")
    print("1. 快速拟合 (仅关键点)")
    print("2. 详细拟合 (关键点 + 点云)")
    
    choice = input("请输入选择 (1/2): ").strip()
    
    if choice == "1":
        measurements, labeled = quick_fit_example()
    elif choice == "2":
        measurements, labeled = detailed_fit_example()
    else:
        print("无效选择，运行快速拟合...")
        measurements, labeled = quick_fit_example()
