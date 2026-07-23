#!/bin/bash
# 快速查看SMPL结果工具

PARAMS_FILE="${1:-outputs/output_from_txt_fixed/smpl_params.npz}"
OUTPUT_HTML="/mnt/d/smpl_body_3d_$(date +%Y%m%d_%H%M%S).html"

echo "正在生成3D可视化..."
python3 -m src.visualization.view_smpl_3d \
    --params "$PARAMS_FILE" \
    --save_html "$OUTPUT_HTML" 2>&1 | grep -v "ERROR\|WARNING"

if [ -f "$OUTPUT_HTML" ]; then
    echo ""
    echo "✓ 成功！"
    echo "✓ HTML文件：$(basename $OUTPUT_HTML)"
    echo "✓ 位置：D:\\"
    echo ""
    echo "在Windows文件管理器中打开 D:\\ 目录"
    echo "双击HTML文件即可在浏览器中查看3D模型"
fi
