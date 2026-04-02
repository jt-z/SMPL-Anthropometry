# SMPL 身体模型拟合使用指南

## 数据准备

你的数据已经准备好了，位于：
```
/home/zjt/dev/On_Git_Projects/3D-Human-Measure/demo3_back_test/output_v2_yolov8-seg(segmenters_yolov8n-seg.pt)/smpl_input.npz
```

数据包含：
- **17个COCO格式的3D关键点** (x, y, z, confidence)，单位：毫米
- **38621个点云数据** (x, y, z)，单位：毫米
- **相机内参**：fx=505.44, fy=505.45, cx=326.83, cy=335.33

## 使用方法

### 方法1：命令行运行（推荐）

```bash
# 基本用法
python fit_smpl_from_data.py

# 指定输入输出路径
python fit_smpl_from_data.py \
    --input /path/to/your/smpl_input.npz \
    --output ./output_results

# 完整参数
python fit_smpl_from_data.py \
    --input /path/to/your/smpl_input.npz \
    --output ./output_results \
    --model_type smpl \
    --gender neutral \
    --keypoint_iterations 300 \
    --pointcloud_iterations 200 \
    --visualize
```

### 方法2：Python脚本调用

```python
from fit_smpl_from_data import SMPLFitterFromData

# 创建拟合器
fitter = SMPLFitterFromData(
    model_path='data',
    model_type='smpl',
    gender='neutral'
)

# 加载数据
npz_path = '/path/to/your/smpl_input.npz'
keypoints_3d, keypoints_valid, pointcloud = fitter.load_data(npz_path)

# 关键点拟合
betas, pose = fitter.fit_to_keypoints(
    keypoints_3d, keypoints_valid,
    num_iterations=300
)

# 身体测量
measurements, labeled_measurements = fitter.measure_body(betas)

# 查看结果
for label, value in labeled_measurements.items():
    print(f"{label}: {value:.2f} cm")

# 可视化
fitter.visualize_results()

# 保存结果
fitter.save_results(
    output_dir='./output',
    betas=betas,
    pose=pose,
    measurements=measurements,
    labeled_measurements=labeled_measurements
)
```

### 方法3：交互式示例

```bash
python example_usage.py
```

选择模式：
1. 快速拟合（仅关键点）
2. 详细拟合（关键点 + 点云）

## 输出结果

### 1. 测量结果

**标准标签格式**：
```
A: 56.32 cm  (头围)
B: 38.45 cm  (颈围)
C: 65.23 cm  (肩到裆高)
D: 90.12 cm  (胸围)
E: 78.56 cm  (腰围)
F: 95.34 cm  (臀围)
...
P: 175.23 cm (身高)
```

**详细名称格式**：
```
height: 175.23 cm
chest_circumference: 90.12 cm
waist_circumference: 78.56 cm
hip_circumference: 95.34 cm
shoulder_breadth: 42.15 cm
...
```

### 2. SMPL参数

```
betas: [1.23, -0.45, 0.67, -0.23, 0.89, 0.12, -0.34, 0.56, -0.12, 0.34]
pose: [0.0, 0.0, 0.0, ...]  (72维姿态参数)
```

### 3. 文件输出

在输出目录中生成：
- `smpl_params.npz`: SMPL参数（betas和pose）
- `measurements.txt`: 测量结果文本文件

### 4. 可视化

运行时添加 `--visualize` 参数，会在浏览器中打开3D可视化界面，显示：
- 人体模型
- 测量位置
- 关键点位置

## 拟合流程

### 步骤1：关键点拟合
- 使用17个COCO关键点约束SMPL模型
- 将COCO关键点映射到SMPL关节点
- 优化形状参数（betas）和姿态参数（pose）

### 步骤2：点云精细拟合（可选）
- 使用点云数据进行精细调整
- 采用ICP风格的优化方法
- 进一步提高拟合精度

### 步骤3：身体测量
- 使用SMPL-Anthropometry库
- 计算16项标准身体测量
- 输出厘米为单位的测量结果

## 参数说明

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` | 自动检测 | 输入npz文件路径 |
| `--output` | `./output_smpl_fit` | 输出目录 |
| `--model_type` | `smpl` | 模型类型（smpl/smplx） |
| `--gender` | `neutral` | 性别（male/female/neutral） |
| `--keypoint_iterations` | `300` | 关键点拟合迭代次数 |
| `--pointcloud_iterations` | `200` | 点云拟合迭代次数 |
| `--visualize` | False | 是否可视化结果 |

### 优化参数调整

**提高精度**：
```bash
python fit_smpl_from_data.py \
    --keypoint_iterations 500 \
    --pointcloud_iterations 300
```

**快速测试**：
```bash
python fit_smpl_from_data.py \
    --keypoint_iterations 100 \
    --pointcloud_iterations 50
```

## 关键点映射

COCO关键点 → SMPL关节点映射：

| COCO索引 | COCO名称 | SMPL索引 | SMPL名称 |
|----------|----------|----------|----------|
| 0 | nose | 15 | head |
| 5 | left_shoulder | 16 | left_shoulder |
| 6 | right_shoulder | 17 | right_shoulder |
| 7 | left_elbow | 18 | left_elbow |
| 8 | right_elbow | 19 | right_elbow |
| 9 | left_wrist | 20 | left_wrist |
| 10 | right_wrist | 21 | right_wrist |
| 11 | left_hip | 1 | left_hip |
| 12 | right_hip | 2 | right_hip |
| 13 | left_knee | 4 | left_knee |
| 14 | right_knee | 5 | right_knee |
| 15 | left_ankle | 7 | left_ankle |
| 16 | right_ankle | 8 | right_ankle |

## 常见问题

### Q1: 拟合精度不高怎么办？
**A**: 增加迭代次数：
```bash
python fit_smpl_from_data.py --keypoint_iterations 500 --pointcloud_iterations 400
```

### Q2: 如何使用SMPLX模型？
**A**: 确保data目录下有SMPLX模型文件，然后：
```bash
python fit_smpl_from_data.py --model_type smplx
```

### Q3: 测量结果单位是什么？
**A**: 所有测量结果都是厘米（cm）。

### Q4: 如何处理多帧数据？
**A**: 编写循环脚本：
```python
import glob
from fit_smpl_from_data import SMPLFitterFromData

fitter = SMPLFitterFromData()

npz_files = glob.glob('/path/to/data/*/smpl_input.npz')
for npz_file in npz_files:
    keypoints_3d, keypoints_valid, pointcloud = fitter.load_data(npz_file)
    betas, pose = fitter.fit_to_keypoints(keypoints_3d, keypoints_valid)
    measurements, labeled = fitter.measure_body(betas)
    # 保存结果...
```

## 性能优化

### GPU加速
确保安装了GPU版本的PyTorch：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 批量处理
对于大量数据，可以使用多进程：
```python
from multiprocessing import Pool

def process_file(npz_file):
    fitter = SMPLFitterFromData()
    # 处理逻辑...

with Pool(4) as p:  # 使用4个进程
    results = p.map(process_file, npz_files)
```

## 下一步

1. **验证结果**：检查测量结果是否合理
2. **调整参数**：根据实际需求调整拟合参数
3. **批量处理**：处理所有帧的数据
4. **结果分析**：统计和分析测量结果

## 技术支持

如有问题，请检查：
1. SMPL模型文件是否正确放置在 `data/smpl/` 或 `data/smplx/` 目录
2. 输入数据格式是否正确
3. Python环境和依赖包是否完整安装
