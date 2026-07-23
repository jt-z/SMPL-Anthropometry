# 🔧 修复PLY格式支持

## 问题诊断

### 错误信息
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/input_points/case1/merged_pointcloud.ply'
```

### 根本原因
1. **文件名错误**：输入了不存在的文件名
2. **代码限制**：`fit_smpl_from_data.py` 只支持NPZ格式，不支持PLY

---

## 快速解决方案

### 方案1：使用正确的文件名（但代码仍需修复）

```bash
# 你的实际文件是：
data/input_points/case1/rail_scan_20260717_014654_replay.ply

# 正确命令：
python3 -m src.fitting.fit_smpl_from_data \
    --input data/input_points/case1/rail_scan_20260717_014654_replay.ply \
    --output outputs/from_rail_scan
```

**注意**：即使文件名正确，代码仍会失败，因为它期望NPZ格式！

---

## 代码问题分析

### 当前代码（fit_smpl_from_data.py第96-100行）

```python
def load_data(self, npz_path):
    data = np.load(npz_path)  # ❌ 假设是NPZ文件
    
    keypoints_3d = data['keypoints_3d']
    keypoints_valid = data['keypoints_valid']
```

**问题**：代码硬编码了NPZ格式，直接用 `np.load()` 读取。

---

## 修复方案

### 需要添加PLY支持

```python
def load_data(self, input_path):
    # 检测文件格式
    ext = os.path.splitext(input_path)[1].lower()
    
    if ext == '.npz':
        # 原有逻辑：读取关键点
        data = np.load(input_path)
        keypoints_3d = data['keypoints_3d']
        keypoints_valid = data['keypoints_valid']
        pointcloud = None
        
    elif ext in ['.ply', '.obj']:
        # 新逻辑：读取点云
        import trimesh
        mesh = trimesh.load(input_path)
        pointcloud = mesh.vertices
        keypoints_3d = None
        keypoints_valid = None
        
    else:
        raise ValueError(f"不支持的文件格式: {ext}")
    
    return keypoints_3d, keypoints_valid, pointcloud
```

---

## 临时解决方案

### 方案A：转换PLY为NPY（快速）

```bash
# 创建转换脚本
python3 << 'EOF'
import trimesh
import numpy as np

# 读取PLY文件
mesh = trimesh.load('data/input_points/case1/rail_scan_20260717_014654_replay.ply')

# 提取顶点
vertices = mesh.vertices

print(f"顶点数: {len(vertices)}")
print(f"范围: X={vertices[:,0].min():.3f}~{vertices[:,0].max():.3f}")
print(f"      Y={vertices[:,1].min():.3f}~{vertices[:,1].max():.3f}")
print(f"      Z={vertices[:,2].min():.3f}~{vertices[:,2].max():.3f}")

# 保存为NPY
np.save('data/input_points/case1/pointcloud.npy', vertices)
print("\n已保存为: data/input_points/case1/pointcloud.npy")
