#!/usr/bin/env python3
"""
点云数据检查工具
用于检查点云文件的格式、大小、范围等信息
"""

import numpy as np
import argparse
import os

def inspect_pointcloud(file_path):
    """检查点云文件"""
    
    print("=" * 60)
    print("           点云数据检查工具")
    print("=" * 60)
    print()
    
    # 1. 文件信息
    print("文件信息:")
    print("-" * 60)
    print(f"  路径: {file_path}")
    print(f"  大小: {os.path.getsize(file_path) / 1024:.2f} KB")
    print(f"  格式: {os.path.splitext(file_path)[1]}")
    print()
    
    # 2. 读取点云
    print("读取点云...")
    print("-" * 60)
    
    try:
        if file_path.endswith('.npy'):
            points = np.load(file_path)
        elif file_path.endswith('.txt'):
            points = np.loadtxt(file_path)
        elif file_path.endswith('.ply'):
            # 简单PLY读取（假设是ASCII格式）
            with open(file_path, 'r') as f:
                lines = f.readlines()
            # 跳过header
            data_start = 0
            for i, line in enumerate(lines):
                if line.startswith('end_header'):
                    data_start = i + 1
                    break
            points = []
            for line in lines[data_start:]:
                parts = line.strip().split()
                if len(parts) >= 3:
                    points.append([float(parts[0]), float(parts[1]), float(parts[2])])
            points = np.array(points)
        else:
            print(f"  ✗ 不支持的格式: {file_path}")
            return
        
        print(f"  ✓ 成功读取")
        print()
        
    except Exception as e:
        print(f"  ✗ 读取失败: {e}")
        return
    
    # 3. 基本统计
    print("点云统计:")
    print("-" * 60)
    print(f"  点数量: {len(points):,}")
    print(f"  数据形状: {points.shape}")
    print()
    
    # 4. 坐标范围
    print("坐标范围:")
    print("-" * 60)
    print(f"  X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]  跨度: {points[:, 0].max() - points[:, 0].min():.3f}")
    print(f"  Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]  跨度: {points[:, 1].max() - points[:, 1].min():.3f}")
    print(f"  Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]  跨度: {points[:, 2].max() - points[:, 2].min():.3f}")
    print()
    
    # 5. 中心点
    center = points.mean(axis=0)
    print("中心点:")
    print("-" * 60)
    print(f"  ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
    print()
    
    # 6. 估计尺度
    print("尺度估计:")
    print("-" * 60)
    height = points[:, 1].max() - points[:, 1].min()  # 假设Y轴向上
    width = points[:, 0].max() - points[:, 0].min()
    depth = points[:, 2].max() - points[:, 2].min()
    
    print(f"  高度: {height:.3f}")
    print(f"  宽度: {width:.3f}")
    print(f"  深度: {depth:.3f}")
    print()
    
    # 7. 单位推测
    print("单位推测:")
    print("-" * 60)
    if height > 100:
        print(f"  可能单位: 毫米 (mm)")
        print(f"  估计身高: {height/10:.1f} cm")
    elif height > 1:
        print(f"  可能单位: 米 (m)")
        print(f"  估计身高: {height*100:.1f} cm")
    else:
        print(f"  可能单位: 未知")
    print()
    
    # 8. 数据质量
    print("数据质量:")
    print("-" * 60)
    
    # 检查NaN
    nan_count = np.isnan(points).sum()
    print(f"  NaN数量: {nan_count}")
    
    # 检查无穷大
    inf_count = np.isinf(points).sum()
    print(f"  Inf数量: {inf_count}")
    
    # 点密度
    density = len(points) / (height * width * depth) if height * width * depth > 0 else 0
    print(f"  点密度: {density:.1f} 点/单位³")
    print()
    
    # 9. 覆盖区域分析
    print("覆盖区域分析:")
    print("-" * 60)
    
    # Y轴（高度）分段统计
    y_bins = np.linspace(points[:, 1].min(), points[:, 1].max(), 10)
    y_hist, _ = np.histogram(points[:, 1], bins=y_bins)
    
    print("  Y轴（高度）分布:")
    for i, count in enumerate(y_hist):
        bar = "█" * int(count / y_hist.max() * 40)
        print(f"    {y_bins[i]:.2f} - {y_bins[i+1]:.2f}: {bar} ({count})")
    print()
    
    # 10. 建议
    print("=" * 60)
    print("建议:")
    print("=" * 60)
    
    if len(points) < 1000:
        print("  ⚠️  点数量较少，可能影响拟合质量")
    elif len(points) > 100000:
        print(f"  ⚠️  点数量很多 ({len(points):,})，建议降采样到20000-50000")
    else:
        print(f"  ✓ 点数量合适 ({len(points):,})")
    
    if nan_count > 0 or inf_count > 0:
        print("  ⚠️  存在异常值，需要清理")
    else:
        print("  ✓ 无异常值")
    
    if height > 100:
        print("  ⚠️  可能需要单位转换 (mm → m)")
    else:
        print("  ✓ 单位合适")
    
    print()
    print("=" * 60)
    print("检查完成！")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description='点云数据检查工具')
    parser.add_argument('--input', type=str, required=True,
                        help='输入点云文件路径 (.npy, .ply, .txt)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"错误: 文件不存在: {args.input}")
        return
    
    inspect_pointcloud(args.input)

if __name__ == '__main__':
    main()
