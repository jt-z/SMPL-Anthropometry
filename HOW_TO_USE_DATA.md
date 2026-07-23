# 📊 如何使用现有数据和运行拟合

**快速指南**

---

## 🎯 当前数据状态

### 现有输出目录

| 目录 | Betas | 质量 | 说明 |
|------|-------|------|------|
| **output_from_txt_fixed/** | ✅ 有值 | ⭐⭐⭐⭐⭐ | **最佳结果，推荐使用** |
| output_frame_1860/ | ❌ 全零 | ⭐⭐ | 默认体型 |
| output_from_txt/ | ❌ 全零 | ⭐⭐ | 旧版本 |
| output_smpl_fit/ | ❌ 全零 | ⭐⭐ | 示例数据 |

### 数据来源

根据代码分析，输入数据来自：
```
/home/zjt/dev/On_Git_Projects/3D-Human-Measure/frame_output/frame_1860_yolo_measure_results.txt
```

这是一个TXT格式的3D关键点文件，包含17个COCO人体关键点。

---

## 🚀 快速开始

### 1. 查看最佳结果

```bash
# 查看测量数据
cat outputs/output_from_txt_fixed/measurements.txt

# 3D可视化
python3 -m src.visualization.view_smpl_3d \
    --params outputs/output_from_txt_fixed/smpl_params.npz

# 保存为HTML
python3 -m src.visualization.view_smpl_3d \
    --params outputs/output_from_txt_fixed/smpl_params.npz \
    --save_html outputs/best_result.html
```

### 2. 查看所有结果对比

```bash
# 使用对比工具
./scripts/compare_results.sh

# 或者批量导出
./scripts/batch_export.sh
```

---

## 📋 输入数据格式

### TXT文件格式（frame_1860_yolo_measure_results.txt）

```
# 17行，每行3个数字（x, y, z坐标）
# 单位：毫米（mm）

# 示例：
x1 y1 z1    # 关键点0: 鼻子
x2 y2 z2    # 关键点1: 左眼
x3 y3 z3    # 关键点2: 右眼
...
x17 y17 z17 # 关键点16: 右踝
```

### 17个COCO关键点

```
0:  鼻子
1:  左眼
2:  右眼
3:  左耳
4:  右耳
5:  左肩
6:  右肩
7:  左肘
8:  右肘
9:  左腕
10: 右腕
11: 左髋
12: 右髋
13: 左膝
14: 右膝
15: 左踝
16: 右踝
```

---

## 🔧 重新运行拟合

### 如果你有原始TXT文件

```bash
# 方法1: 使用修复版（推荐）
python3 -m src.fitting.fit_smpl_from_txt_fixed \
    --input /path/to/frame_1860_yolo_measure_results.txt \
    --output outputs/new_result \
    --keypoint_iterations 500 \
    --visualize

# 方法2: 使用原始版
python3 -m src.fitting.fit_smpl_from_txt \
    --input /path/to/frame_1860_yolo_measure_results.txt \
    --output outputs/test_result
```

### 如果你有新的关键点数据

```bash
# 准备你的TXT文件（17行x3列）
# 格式: x y z (每行一个关键点)

# 运行拟合
python3 -m src.fitting.fit_smpl_from_txt_fixed \
    --input your_keypoints.txt \
    --output outputs/your_result \
    --visualize
```

---

## 📊 查看和分析结果

### 查看测量结果

```bash
# 查看文本结果
cat outputs/output_from_txt_fixed/measurements.txt

# 提取关键指标
grep -E "height|chest|waist|hip" outputs/output_from_txt_fixed/measurements.txt
```

### 查看SMPL参数

```bash
# 使用Python查看
python3 << 'PYEOF'
import numpy as np
data = np.load('outputs/output_from_txt_fixed/smpl_params.npz')
print("Betas (shape parameters):")
print(data['betas'])
print(f"\n统计:")
print(f"  均值: {data['betas'].mean():.3f}")
print(f"  标准差: {data['betas'].std():.3f}")
PYEOF
```

### 3D可视化

```bash
# 交互式查看
python3 -m src.visualization.view_smpl_3d \
    --params outputs/output_from_txt_fixed/smpl_params.npz

# 保存HTML（可离线查看）
python3 -m src.visualization.view_smpl_3d \
    --params outputs/output_from_txt_fixed/smpl_params.npz \
    --save_html /mnt/d/result.html

# 只看模型（不显示测量线）
python3 -m src.visualization.view_smpl_3d \
    --params outputs/output_from_txt_fixed/smpl_params.npz \
    --no_measurements
```

---

## 🎯 推荐工作流程

### 场景1: 分析现有结果

```bash
# 1. 查看对比
./scripts/compare_results.sh

# 2. 查看最佳结果
cat outputs/output_from_txt_fixed/measurements.txt

# 3. 3D可视化
python3 -m src.visualization.view_smpl_3d \
    --params outputs/output_from_txt_fixed/smpl_params.npz \
    --save_html outputs/best.html
```

### 场景2: 处理新数据

```bash
# 1. 准备TXT文件（17行x3列）
# 2. 运行拟合
python3 -m src.fitting.fit_smpl_from_txt_fixed \
    --input new_data.txt \
    --output outputs/new_result \
    --visualize

# 3. 查看结果
cat outputs/new_result/measurements.txt
```

### 场景3: 批量处理

```bash
# 批量导出所有结果为HTML
./scripts/batch_export.sh

# 结果保存在 outputs/*.html
# 可以在浏览器中打开查看
```

---

## 🔍 诊断和调试

### 检查输入数据

```bash
# 查看TXT文件格式
head -20 your_data.txt

# 统计行数（应该是17）
wc -l your_data.txt

# 检查数据范围
python3 << 'PYEOF'
import numpy as np
data = np.loadtxt('your_data.txt')
print(f"Shape: {data.shape}")  # 应该是 (17, 3)
print(f"Min: {data.min():.3f}")
print(f"Max: {data.max():.3f}")
print(f"Mean: {data.mean():.3f}")
PYEOF
```

### 诊断关键点

```bash
# 使用诊断工具
python3 tools/diagnose_keypoints.py
```

### 评估拟合质量

```bash
# 对比两个结果
python3 tools/evaluate.py \
    --result1 outputs/output_from_txt_fixed/ \
    --result2 outputs/output_from_txt/
```

---

## 💡 常见问题

### Q1: 为什么有些结果Betas全是0？
**A:** Betas全零表示使用的是平均体型，没有实际拟合。只有 `output_from_txt_fixed` 有真实拟合结果。

### Q2: 如何找到原始TXT文件？
**A:** 根据代码，原始文件在：
```
/home/zjt/dev/On_Git_Projects/3D-Human-Measure/frame_output/frame_1860_yolo_measure_results.txt
```

### Q3: TXT文件格式是什么？
**A:** 17行，每行3个数字（x y z），表示17个COCO关键点的3D坐标。

### Q4: 如何获得更好的拟合效果？
**A:** 
1. 使用 `fit_smpl_from_txt_fixed.py`（已修复坐标系和单位）
2. 增加迭代次数：`--keypoint_iterations 1000`
3. 确保输入关键点质量高

### Q5: 如何在Windows中查看3D结果？
**A:** 
```bash
# 保存HTML到D盘
python3 -m src.visualization.view_smpl_3d \
    --params outputs/output_from_txt_fixed/smpl_params.npz \
    --save_html /mnt/d/result.html

# 然后在Windows文件管理器中打开 D:\result.html
```

---

## 📖 相关文档

- **TXT拟合指南**: [docs/TXT_FITTING_GUIDE.md](docs/TXT_FITTING_GUIDE.md)
- **数据来源分析**: [DATA_SOURCE_ANALYSIS.md](DATA_SOURCE_ANALYSIS.md)
- **使用指南**: [docs/USAGE_GUIDE.md](docs/USAGE_GUIDE.md)
- **工具脚本**: [docs/README_TOOLS.md](docs/README_TOOLS.md)

---

## 🎉 快速命令

```bash
# 查看最佳结果
cat outputs/output_from_txt_fixed/measurements.txt

# 3D可视化
./scripts/quick_view.sh outputs/output_from_txt_fixed/smpl_params.npz

# 对比所有结果
./scripts/compare_results.sh

# 批量导出HTML
./scripts/batch_export.sh
```

---

**创建时间:** 2026-07-24  
**数据位置:** outputs/output_from_txt_fixed/ ⭐
