"""
可视化 SMPL 体型测量结果。

用法:
    python visualize_measurements.py
    python visualize_measurements.py --input ./fit_output/measurements.txt
    python visualize_measurements.py --input ./fit_output/measurements.txt --save ./fit_output/measurements_vis.png
"""

import argparse
import re
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']


# ── 分组定义 ────────────────────────────────────────────
GROUPS = {
    "Height / Length": {
        "color": "#4C9BE8",
        "items": [
            "height",
            "crotch height",
            "inside leg height",
            "shoulder to crotch height",
            "arm left length",
            "arm right length",
            "arm length (shoulder to elbow)",
            "arm length (spine to wrist)",
        ],
    },
    "Circumference": {
        "color": "#E87B4C",
        "items": [
            "head circumference",
            "neck circumference",
            "chest circumference",
            "waist circumference",
            "hip circumference",
            "Hip circumference max height",
            "thigh left circumference",
            "calf left circumference",
            "bicep right circumference",
            "forearm right circumference",
            "wrist right circumference",
            "ankle left circumference",
        ],
    },
    "Breadth": {
        "color": "#5DBE7A",
        "items": [
            "shoulder breadth",
        ],
    },
}

# 显示名称映射（截短 / 美化）
DISPLAY_NAMES = {
    "Hip circumference max height":    "Hip circ. (max height)",
    "arm length (shoulder to elbow)":  "Arm (shoulder→elbow)",
    "arm length (spine to wrist)":     "Arm (spine→wrist)",
    "shoulder to crotch height":       "Shoulder→crotch",
}


def parse_measurements(filepath):
    measurements = {}
    gender = "NEUTRAL"
    betas_str = ""

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("性别:"):
                gender = line.split(":", 1)[1].strip()
            elif line.startswith("betas:"):
                betas_str = line.split(":", 1)[1].strip()
            else:
                m = re.match(r"^(.+?):\s*([\d.]+)\s*$", line)
                if m:
                    name = m.group(1).strip()
                    value = float(m.group(2))
                    measurements[name] = value

    return measurements, gender, betas_str


def build_ordered_data(measurements):
    """按分组顺序整理数据，未归组的放最后"""
    rows = []   # (display_name, value, color, group_name)

    assigned = set()
    for group_name, group in GROUPS.items():
        color = group["color"]
        for item in group["items"]:
            if item in measurements:
                dname = DISPLAY_NAMES.get(item, item.title())
                rows.append((dname, measurements[item], color, group_name))
                assigned.add(item)

    # 未分组的
    for name, val in measurements.items():
        if name not in assigned:
            dname = DISPLAY_NAMES.get(name, name.title())
            rows.append((dname, val, "#AAAAAA", "Other"))

    return rows


def draw_figure(measurements, gender, betas_str, save_path=None):
    rows = build_ordered_data(measurements)
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colors = [r[2] for r in rows]

    n = len(rows)
    fig_h = max(6, n * 0.45 + 2.5)
    fig, ax = plt.subplots(figsize=(12, fig_h))

    y_pos = np.arange(n)
    bars = ax.barh(y_pos, values, color=colors, height=0.65, edgecolor="white", linewidth=0.5)

    # 数值标注
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f} cm", va="center", ha="left", fontsize=9, color="#333333")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("cm", fontsize=11)
    ax.set_xlim(0, max(values) * 1.18)
    ax.set_title(f"SMPL Body Measurements  |  Gender: {gender}",
                 fontsize=13, fontweight="bold", pad=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    # 分组分隔线 + 图例
    legend_patches = []
    seen_groups = {}
    prev_group = None
    for i, (_, _, color, group_name) in enumerate(rows):
        if group_name not in seen_groups:
            seen_groups[group_name] = color
            legend_patches.append(mpatches.Patch(color=color, label=group_name))
        if group_name != prev_group and i > 0:
            ax.axhline(y=i - 0.5, color="#CCCCCC", linewidth=0.8, linestyle="-")
        prev_group = group_name

    ax.legend(handles=legend_patches, loc="lower right", fontsize=9,
              framealpha=0.85, edgecolor="#CCCCCC")

    # betas 注释
    if betas_str:
        fig.text(0.01, 0.005, f"betas: {betas_str}",
                 fontsize=7, color="#999999", va="bottom")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"图表已保存: {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="可视化 SMPL 体型测量结果")
    parser.add_argument("--input", default="./fit_output/measurements.txt",
                        help="measurements.txt 路径")
    parser.add_argument("--save", default=None,
                        help="保存图片路径 (如 ./fit_output/measurements_vis.png)，"
                             "不指定则弹出窗口")
    args = parser.parse_args()

    measurements, gender, betas_str = parse_measurements(args.input)
    print(f"读取到 {len(measurements)} 项测量值 | 性别: {gender}")

    draw_figure(measurements, gender, betas_str, save_path=args.save)


if __name__ == "__main__":
    main()
