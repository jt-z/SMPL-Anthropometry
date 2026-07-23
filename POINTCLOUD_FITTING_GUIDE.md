# 🔧 点云SMPL拟合实践指南

**基于你的数据：** rail_scan_20260717_014654_replay.ply

---

## 📊 你的数据分析

### 文件信息
```
文件: rail_scan_20260717_014654_replay.ply
大小: 705.69 KB
格式: PLY (二进制，小端序)

OBJ文件: 2095.97 KB
顶点: ~13,000个
面: 有网格数据
```

### 数据特征
- ✅ 有PLY和OBJ两种格式
- ✅ 包含网格数据（不仅是点云）
- ✅ 文件大小合适
- 📅 采集时间：2026-07-17 01:46

---

## 🚀 快速开始

### 方法1：使用测试脚本（推荐）

```bash
# 运行自动化测试
./test_pointcloud_fitting.sh
```

这个脚本会：
1. 检查文件和依赖
2. 运行SMPL拟合
3. 显示测量结果
4. 提供可视化命令

### 方法2：手动运行

```bash
# 1. 检查点云数据
python3 tools/inspect_pointcloud.py \
    --input data/input_points/case1/rail_scan_20260717_014654_replay.ply

# 2. 运行拟合
python3 -m src.fitting.fit_smpl_from_data \
    --input data/input_points/case1/rail_scan_20260717_014654_replay.ply \
    --output outputs/from_rail_scan \
    --visualize

# 3. 查看结果
cat outputs/from_rail_scan/measurements.txt

# 4. 3D可视化
python3 -m src.visualization.view_smpl_3d \
    --params outputs/from_rail_scan/smpl_params.npz
```

---

## 📋 预期结果

### 可能的情况

#### 情况1：拟合成功 ✅
**表现：**
- Betas参数不全为零
- 测量值合理（身高150-200cm）
- 可视化效果自然

**下一步：**
- 验证关键测量项（胸围、腰围）
- 与实际测量对比
- 决定是否需要改进

#### 情况2：拟合一般 ⚠️
**表现：**
- Betas有值但效果不理想
- 某些测量项不准确
- 可视化有些奇怪

**下一步：**
- 调整优化参数
- 增加迭代次数
- 考虑预处理点云

#### 情况3：拟合失败 ❌
**表现：**
- Betas全零或异常
- 测量值明显错误
- 程序报错

**原因可能：**
- 坐标系不匹配
- 单位问题（mm vs m）
- 点云质量问题
- 数据太不完整

**解决方案：**
- 检查并转换坐标系
- 调整单位
- 预处理点云
- 开发改进版本

---

## 🔍 结果验证

### 1. 检查Betas参数

```bash
python3 << 'EOF'
import numpy as np
data = np.load('outputs/from_rail_scan/smpl_params.npz')
betas = data['betas']

print(f"Betas: {betas}")
print(f"均值: {betas.mean():.3f}")
print(f"标准差: {betas.std():.3f}")
print(f"是否全零: {np.allclose(betas, 0)}")

if np.allclose(betas, 0):
    print("\n⚠️  拟合可能失败（参数全零）")
else:
    print("\n✓ 拟合成功（参数有值）")
