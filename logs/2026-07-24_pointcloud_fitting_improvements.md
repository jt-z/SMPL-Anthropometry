# 2026-07-24 点云拟合改进日志

## 问题背景

用户使用不完整的人体点云数据（只有前侧）拟合SMPL模型，发现拟合效果很差：
- 点云和SMPL位置完全错位
- 优化迭代只跑1次就停止
- 拟合结果SMPL方向错误（背对着点云）

测试数据：`/home/zjt/dev/on_git/SMPL-Anthropometry/data/input_points/case1/rail_scan_20260717_014654_replay.ply`
- 点云点数：14164
- 坐标范围：X[-0.317, 0.327], Y[-0.447, 0.532], Z[0.699, 0.900]（米）

---

## 修复的关键Bug

### 1. **梯度断开问题**（最严重）

**问题定位：**
- 在 `fit_to_pointcloud()` 的优化循环中，调用 `get_smpl_joints()` 返回numpy数组
- 再用 `torch.tensor()` 转回tensor，导致梯度链断开
- 优化器无法反向传播，参数无法更新

**修复方案：**
```python
# 修复前（错误）
vertices, joints = self.get_smpl_joints(betas, pose)  # 返回numpy
vertices_torch = torch.tensor(vertices, ...)  # 无梯度

# 修复后（正确）
output = self.model(
    betas=betas.unsqueeze(0),
    body_pose=pose[:, 3:],
    global_orient=pose[:, :3],
    return_verts=True
)
vertices_torch = output.vertices[0]  # 保持梯度
```

**效果：**
- 修复前：迭代只跑1次，损失不变
- 修复后：正常迭代200次，损失从0.006885降到0.000052

---

### 2. **缺少平移参数**

**问题定位：**
- SMPL模型位置固定在原点附近
- 点云中心在 [0.015, 0.059, 0.791]（米）
- 两者空间位置不匹配，无法对齐

**修复方案：**
```python
# 添加可优化的平移参数
pc_center = sampled_points.mean(axis=0)
transl = torch.tensor(pc_center, dtype=torch.float32, requires_grad=True, device=self.device)

# 在模型输出上加平移
vertices_torch = output.vertices[0] + transl

# 加入优化器
optimizer = torch.optim.Adam([betas, pose, transl], lr=0.005)
```

**效果：**
- 损失从0.000052进一步降到0.000001
- SMPL可以自动移动到点云位置

---

### 3. **单位换算错误**

**问题定位：**
- 代码假设点云单位是毫米，执行 `/1000.0` 转换为米
- 实际点云已经是米单位
- 导致点云被缩小1000倍，成为0.79毫米的微小物体

**修复方案：**
```python
# 修复前（错误）
target_points_m = sampled_points / 1000.0

# 修复后（正确）
target_torch = torch.tensor(sampled_points, dtype=torch.float32, device=self.device)
```

**效果：**
- 点云和SMPL在相同的空间尺度
- 拟合可以正常进行

---

## 新增功能

### 1. **姿态冻结模式**

**问题：**
不完整点云（只有前侧）导致SMPL找到错误的局部最优解——让背部去拟合点云前部。

**解决方案：**
```python
def fit_to_pointcloud(self, pointcloud, freeze_pose=True, ...):
    if freeze_pose:
        # 只优化body shape和位置，姿态固定
        pose.requires_grad = False
        optimizer = torch.optim.Adam([betas, transl], lr=0.01)
    else:
        # 优化所有参数
        optimizer = torch.optim.Adam([betas, pose, transl], lr=0.005)
```

**命令行参数：**
```bash
# 默认冻结姿态（推荐用于不完整点云）
--freeze_pose

# 解冻姿态（用于完整点云）
--optimize_pose
```

---

### 2. **初始姿态设置**

为匹配真实人体姿态，添加了手臂角度调整：

```python
# 自然站姿：手臂下垂
pose = torch.zeros(72, dtype=torch.float32, device=self.device)

# 左肩：向下旋转45度
pose[40] = 0.7854  # π/4 rad
# 右肩：向下旋转45度
pose[43] = -0.7854
```

---

### 3. **可视化工具**

#### 工具1：对比点云生成器
`tools/create_comparison_pointclouds.py`

**功能：**
- 从SMPL网格表面均匀采样点云（默认10000点）
- 生成蓝色原始点云 + 红色SMPL点云
- 输出合并文件用于对比

**使用：**
```bash
python3 tools/create_comparison_pointclouds.py \
  --pointcloud data/input_points/case1/rail_scan.ply \
  --params output/smpl_params.npz \
  --output_dir output \
  --smpl_samples 10000
```

**输出文件：**
- `original_blue.ply` - 原始点云（蓝色）
- `smpl_red.ply` - SMPL采样点云（红色）
- `comparison.ply` - 合并点云

#### 工具2：手臂姿态测试
`tools/test_arm_poses.py`

**功能：**
生成不同手臂角度的SMPL模型（0°-90°），用于找到最佳初始姿态

**使用：**
```bash
python3 tools/test_arm_poses.py
```

**输出：**
- `smpl_arm_poses/smpl_arms_0deg.ply` - T-pose（手臂水平）
- `smpl_arm_poses/smpl_arms_45deg.ply` - 中等下垂
- `smpl_arm_poses/smpl_arms_90deg.ply` - 完全垂直

---

## 改进效果对比

| 指标 | 修复前 | 修复后（无平移） | 修复后（有平移） | 冻结姿态 |
|------|--------|------------------|------------------|----------|
| 迭代次数 | 1次 | 200次 | 200次 | 300次 |
| 最终损失 | 0.006885 | 0.000052 | 0.000001 | 0.001741 |
| 点云-SMPL中心距离 | N/A | 0.142m | 0.024m | 0.238m |
| 方向正确性 | 错误 | 错误（背对） | 错误（背对） | 正确 |
| 手臂姿态 | T-pose | T-pose | T-pose | 可调整 |

---

## 当前问题

### 1. **初始姿态未对准**
- 点云中人体的真实姿态未知（站立？躺着？手臂角度？）
- SMPL默认是T-pose（手臂水平展开）
- 需要手动调整初始姿态来匹配点云

### 2. **不完整点云限制**
- 只有前侧点云，缺少背部数据
- 冻结姿态后损失较高（0.001741 vs 0.000001）
- 背部的拟合精度无法验证

### 3. **缺少旋转对齐**
- 如果点云和SMPL朝向不同（如点云躺着），需要初始旋转对齐
- 当前只有平移对齐，没有旋转对齐

---

## 下一步工作

1. **确定点云真实姿态**
   - 在CloudCompare中观察点云
   - 使用 `test_arm_poses.py` 找到最接近的手臂角度
   - 确认是站立还是躺姿

2. **添加旋转对齐**
   - 如果点云是躺着的，添加global_orient优化
   - 或者手动设置初始旋转矩阵

3. **改进优化策略**
   - 对不完整区域降低权重
   - 添加对称性约束（左右对称）
   - 使用分阶段优化（先粗调位置，再精调shape）

---

## 代码修改文件

### 核心文件
- `src/fitting/fit_smpl_from_data.py` - 主要拟合逻辑
  - 修复梯度断开（行249-308）
  - 添加平移参数优化
  - 添加姿态冻结模式
  - 修正单位换算

### 新增工具
- `tools/visualize_fit.py` - 可视化拟合结果
- `tools/create_comparison_pointclouds.py` - 生成对比点云
- `tools/test_arm_poses.py` - 测试不同手臂角度

### 输出示例
- `output_case1_correct_unit/` - 修复单位后的结果
- `output_case1_frozen_pose/` - 冻结姿态的结果
- `output_case1_arms_down/` - 手臂下垂姿态的结果
- `smpl_arm_poses/` - 不同角度的SMPL参考模型

---

## 使用建议

### 对于不完整点云（推荐）
```bash
python3 src/fitting/fit_smpl_from_data.py \
  --input data/input_points/case1/scan.ply \
  --output output/result \
  --freeze_pose \
  --pointcloud_iterations 300
```

### 对于完整点云
```bash
python3 src/fitting/fit_smpl_from_data.py \
  --input data/input_points/full_scan.ply \
  --output output/result \
  --optimize_pose \
  --pointcloud_iterations 500
```

### 可视化结果
```bash
python3 tools/create_comparison_pointclouds.py \
  --pointcloud data/input_points/case1/scan.ply \
  --params output/result/smpl_params.npz \
  --output_dir output/result \
  --smpl_samples 10000
```

---

## 测量结果

冻结姿态拟合后的身体测量（case1）：

```
身高(P):        171.42 cm
胸围(D):        102.63 cm
腰围(E):         91.60 cm
臀围(F):        103.12 cm
肩宽(A):         55.39 cm
```

---

## 技术要点总结

1. **保持梯度流** - 在优化循环中必须使用完整的torch计算图
2. **平移参数必不可少** - SMPL默认在原点，必须添加平移才能对齐任意位置的点云
3. **姿态约束很重要** - 不完整点云必须冻结姿态，避免错误的局部最优
4. **单位要一致** - 确认点云和SMPL模型的单位（米 vs 毫米）
5. **初始姿态影响大** - 手臂角度、站姿/躺姿等初始设置会影响最终结果

---

**日志作者：** Claude (Kiro)  
**日期：** 2026-07-24  
**项目：** SMPL-Anthropometry  
