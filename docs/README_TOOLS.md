# 🛠️ 工具任务使用指南

本项目已为你创建了多个实用工具脚本，方便日常使用。

## 📋 可用工具

### 1. quick_view.sh - 快速查看工具
快速生成3D可视化HTML文件

```bash
# 查看默认结果
./quick_view.sh

# 查看指定结果
./quick_view.sh outputs/output_from_txt_fixed/smpl_params.npz
```

### 2. batch_export.sh - 批量导出
将所有SMPL结果批量导出为HTML文件

```bash
./batch_export.sh
```

### 3. compare_results.sh - 结果对比
对比不同结果的关键测量数据

```bash
./compare_results.sh
```

### 4. quick_commands.sh - 快捷命令
提供常用命令的快捷方式

```bash
# 查看帮助
./quick_commands.sh

# 3D查看
./quick_commands.sh view outputs/output_from_txt_fixed/smpl_params.npz

# 导出HTML
./quick_commands.sh export outputs/output_from_txt_fixed/smpl_params.npz

# 显示测量数据
./quick_commands.sh measure outputs/output_from_txt_fixed/smpl_params.npz

# 对比所有结果
./quick_commands.sh compare

# 列出所有输出
./quick_commands.sh list

# 检查模型文件
./quick_commands.sh check
```

## 🎯 常见任务

### 任务1：查看现有结果
```bash
# 方式1：使用快捷命令
./quick_commands.sh view outputs/output_from_txt_fixed/smpl_params.npz

# 方式2：直接命令
python3 -m src.visualization.view_smpl_3d \
    --params outputs/output_from_txt_fixed/smpl_params.npz
```

### 任务2：导出为离线HTML
```bash
# 方式1：使用快捷命令（自动保存到D盘）
./quick_commands.sh export outputs/output_from_txt_fixed/smpl_params.npz

# 方式2：使用quick_view（自动命名）
./quick_view.sh outputs/output_from_txt_fixed/smpl_params.npz

# 方式3：直接命令（自定义路径）
python3 -m src.visualization.view_smpl_3d \
    --params outputs/output_from_txt_fixed/smpl_params.npz \
    --save_html /mnt/d/my_result.html
```

### 任务3：批量处理所有结果
```bash
./batch_export.sh
```

### 任务4：对比不同结果
```bash
./compare_results.sh
```

### 任务5：只查看测量数据
```bash
cat outputs/output_from_txt_fixed/measurements.txt

# 或使用快捷命令
./quick_commands.sh measure outputs/output_from_txt_fixed/smpl_params.npz
```

## 📂 文件位置

### WSL路径 → Windows路径
- 项目根目录：`\\wsl$\Ubuntu\home\zjt\dev\on_git\SMPL-Anthropometry`
- 输出目录：`\\wsl$\Ubuntu\home\zjt\dev\on_git\SMPL-Anthropometry\outputs`
- D盘HTML：`D:\smpl_body_3d_*.html`

### 在Windows中打开
1. 打开文件管理器
2. 地址栏输入：`\\wsl$\Ubuntu\home\zjt\dev\on_git\SMPL-Anthropometry\outputs`
3. 双击HTML文件即可在浏览器中查看

## 🔧 解决浏览器问题

如果在WSL2中打开浏览器有问题，推荐使用以下方法：

```bash
# 保存HTML到D盘
python3 -m src.visualization.view_smpl_3d \
    --params outputs/output_from_txt_fixed/smpl_params.npz \
    --save_html /mnt/d/body_3d.html

# 然后在Windows文件管理器中打开 D:\body_3d.html
```

## 📊 数据分析

### 查看SMPL参数
```bash
python3 -c "import numpy as np; d=np.load('outputs/output_from_txt_fixed/smpl_params.npz'); print('Betas:', d['betas'])"
```

### 提取关键测量
```bash
grep -E "height|chest|waist|hip|shoulder" \
    outputs/output_from_txt_fixed/measurements.txt
```

### 对比多个结果
```bash
for dir in outputs/output_*/; do
    echo "=== $(basename $dir) ==="
    grep "height " "$dir/measurements.txt" 2>/dev/null
done
```

## ✅ 验证工具

### 检查所有模块
```bash
python3 -c "from src.core.measure import MeasureBody; print('✓ core')"
python3 -c "from src.fitting.fit_smpl_from_data import SMPLFitterFromData; print('✓ fitting')"
```

### 列出所有输出
```bash
./quick_commands.sh list
```

### 检查SMPL模型
```bash
python3 tools/check_models.py
```

## 🎉 快速开始

```bash
# 1. 查看所有可用结果
./quick_commands.sh list

# 2. 查看测量数据
cat outputs/output_from_txt_fixed/measurements.txt

# 3. 导出3D可视化
./quick_view.sh

# 4. 在Windows中打开
# 文件位置：D:\smpl_body_3d_*.html
```

---

**提示：** 所有工具脚本都已经创建好了，直接运行即可！
