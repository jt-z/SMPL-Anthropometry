#!/bin/bash
# 批量导出所有结果为HTML

echo "正在批量导出SMPL结果为HTML..."
echo ""

count=0
for dir in outputs/output_*/; do
  if [ -f "$dir/smpl_params.npz" ]; then
    name=$(basename "$dir")
    output_file="outputs/${name}_3d.html"
    
    echo "处理: $name"
    python3 -m src.visualization.view_smpl_3d \
      --params "$dir/smpl_params.npz" \
      --no_measurements \
      --save_html "$output_file" 2>&1 | grep -v "ERROR\|WARNING" | grep "保存"
    
    if [ -f "$output_file" ]; then
      size=$(ls -lh "$output_file" | awk '{print $5}')
      echo "  ✓ 已生成 $output_file ($size)"
      count=$((count+1))
    fi
    echo ""
  fi
done

echo "======================================"
echo "完成！共生成 $count 个HTML文件"
echo "======================================"
