#!/bin/bash
# 对比不同结果的测量数据

echo "====================================="
echo "        测量结果对比"
echo "====================================="
echo ""

for dir in outputs/output_*/; do
  if [ -f "$dir/measurements.txt" ]; then
    name=$(basename "$dir")
    echo "[$name]"
    echo "----------------------------------------"
    grep -E "height |chest |waist |hip |shoulder breadth" "$dir/measurements.txt" | \
      sed 's/^/  /'
    echo ""
  fi
done
