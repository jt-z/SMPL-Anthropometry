# 工具脚本说明

本目录包含项目的实用工具脚本。

## 📋 可用脚本

### 1. quickstart.sh - 快速测试
快速测试项目功能是否正常

```bash
cd /path/to/SMPL-Anthropometry
./scripts/quickstart.sh
```

### 2. quick_view.sh - 快速查看
快速生成3D可视化并导出到D盘

```bash
./scripts/quick_view.sh [smpl_params.npz路径]
```

### 3. batch_export.sh - 批量导出
将所有SMPL结果批量导出为HTML文件

```bash
./scripts/batch_export.sh
```

### 4. compare_results.sh - 结果对比
对比不同结果的测量数据

```bash
./scripts/compare_results.sh
```

### 5. quick_commands.sh - 快捷命令
提供常用命令的快捷方式

```bash
./scripts/quick_commands.sh <command>

可用命令:
  view <file>      - 3D查看
  export <file>    - 导出HTML
  measure <file>   - 显示测量数据
  compare          - 对比所有结果
  list             - 列出所有输出
  check            - 检查模型文件
```

### 6. push_to_remote.sh - 推送到远程
带确认提示的Git推送脚本

```bash
./scripts/push_to_remote.sh
```

## 💡 使用技巧

### 从项目根目录运行
```bash
# 快速查看
./scripts/quick_view.sh

# 批量导出
./scripts/batch_export.sh
```

### 创建快捷方式（可选）
```bash
# 在根目录创建软链接
ln -s scripts/quick_view.sh quick_view
ln -s scripts/batch_export.sh batch_export
```

## 📖 更多信息

详细使用说明请查看：`docs/README_TOOLS.md`
