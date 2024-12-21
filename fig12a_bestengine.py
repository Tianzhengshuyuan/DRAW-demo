import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
import os

# 数据示例：矩阵内容定义（每个单元格对应的标签）
data = [['CPU', 'EF2','EF2', 'SWF','SWF', 'EMO', 'EMO', 'ENX','ENX', 'MV2','MV2', 'MV','MV', 'LVT','LVT'],
        ['RWC','TFLIite','OpenVINO','OpenVINO','OpenVINO','TFLIite','OpenVINO','MNN','OpenVINO','TFLIite','OpenVINO','OpenVINO','OpenVINO','ONNXRT','OpenVINO'],
        ['CMT','MNN','OpenVINO','MNN','OpenVINO','TFLIite','OpenVINO','MNN','MNN','ONNXRT','TFLIite','ONNXRT','OpenVINO','MNN','OpenVINO'],
        ['LNC','TFLIite','OpenVINO','OpenVINO','OpenVINO','TFLIite','OpenVINO','MNN','OpenVINO','TFLIite','TFLIite','OpenVINO','OpenVINO','OpenVINO','OpenVINO'],
        ['SMT','MNN','OpenVINO','MNN','OpenVINO','TFLIite','OpenVINO','MNN','OpenVINO','TFLIite','TFLIite','MNN','OpenVINO','MNN','OpenVINO'],
        ['M1P','MNN','MNN','MNN','MNN','TFLIite','MNN','MNN','TFLIite','ncnn','ncnn','TFLIite','MNN','MNN','MNN'],
        ['M1E','TFLIite','MNN','TFLIite','TFLIite','TFLIite','TFLIite','MNN','MNN','ncnn','ncnn','TFLIite','MNN','MNN','MNN'],
        ['A71x','TFLIite','MNN','TFLIite','TFLIite','TFLIite','TFLIite','MNN','MNN','ncnn','ncnn','TFLIite','MNN','TFLIite','MNN'],
        ['A78','TFLIite','MNN','TFLIite','TFLIite','TFLIite','TFLIite','MNN','MNN','ncnn','ncnn','TFLIite','MNN','MNN','MNN'],
        ['A76','TFLIite','MNN','TFLIite','MNN','TFLIite','TFLIite','MNN','MNN','ncnn','ncnn','TFLIite','MNN','MNN','MNN'],
        ['A55','TFLIite','MNN','TFLIite','TFLIite','TFLIite','TFLIite','MNN','MNN','ncnn','ncnn','TFLIite','TFLIite','TFLIite','MNN'],
        ['A73','MNN','TFLIite','TFLIite','TFLIite','TFLIite','TFLIite','TFLIite','TFLIite','TFLIite','TFLIite','TFLIite','TFLIite','MNN','TFLIite'],
        ['A72','TFLIite','MNN','TFLIite','MNN','TFLIite','MNN','TFLIite','MNN','ncnn','ncnn','TFLIite','ONNXRT','MNN','TFLIite'],
        ['M5','TFLIite','empty','TFLIite','empty','TFLIite','empty','MNN','empty','TFLIite','empty','TFLIite','empty','TFLIite','empty'],
        ['X3','TFLIite','empty','MNN','empty','TFLIite','empty','MNN','empty','ncnn','empty','TFLIite','empty','MNN','empty'],
        ['X1','TFLIite','empty','MNN','empty','TFLIite','empty','MNN','empty','TFLIite','empty','TFLIite','empty','MNN','empty'],
        ['A75','TFLIite','empty','TFLIite','empty','TFLIite','empty','MNN','empty','TFLIite','empty','TFLIite','empty','TFLIite','empty'],
        ['T200','TFLIite','empty','MNN','empty','TFLIite','empty','MNN','empty','ncnn','empty','TFLIite','empty','MNN','empty'],
        ['A510','TFLIite','empty','TFLIite','empty','TFLIite','empty','MNN','empty','TFLIite','empty','TFLIite','empty','TFLIite','empty'],
        ['A53','TFLIite','empty','TFLIite','empty','TFLIite','empty','TFLIite','empty','ncnn','empty','TFLIite','empty','TFLIite','empty']
]


# 颜色映射
color_map = {
    'TFLIite': '#5491f4',   # 蓝色
    'MNN': '#ec5548',      # 红色
    'ncnn': '#33a852',     # 绿色
    'ONNXRT': '#ff9632',   # 橘色
    'OpenVINO': '#87CEFA',     # 浅蓝色
    'empty': '#b6b6b6'
}

fig, ax = plt.subplots(figsize=(12, 6))

# 绘制网格
n_rows = len(data)
n_cols = len(data[1])

for i in range(n_rows):
    for j in range(n_cols):
        value = data[i][j]   # 空单元格填充灰色
        rect = patches.Rectangle(
            (j, n_rows - i - 1),  # 坐标 (x, y)
            1, 1,                 # 宽度和高度
            facecolor=color_map.get(value, '#FFFFFF'),  # 根据值填充颜色
            edgecolor='black',     # 设置边框颜色
            linewidth=1
        )
        if i > 0:
            if value not in color_map:
                ax.text(
                    j + 0.5, n_rows - i - 0.5,  # 坐标中心
                    value,
                    ha='center', va='center', fontsize=16, color='black'
                )
        ax.add_patch(rect)

ax.add_line(lines.Line2D([2, 2], [19, 20], color='white', linewidth=2))
ax.add_line(lines.Line2D([4, 4], [19, 20], color='white', linewidth=2))
ax.add_line(lines.Line2D([6, 6], [19, 20], color='white', linewidth=2))
ax.add_line(lines.Line2D([8, 8], [19, 20], color='white', linewidth=2))
ax.add_line(lines.Line2D([10, 10], [19, 20], color='white', linewidth=2))
ax.add_line(lines.Line2D([12, 12], [19, 20], color='white', linewidth=2))
ax.add_line(lines.Line2D([14, 14], [19, 20], color='white', linewidth=2))

ax.text(0.5, 19.5, 'CPU', ha='center', va='center', fontsize=16, color='black')
ax.text(2, 19.5, 'EF2', ha='center', va='center', fontsize=16, color='black')
ax.text(4, 19.5, 'SWF', ha='center', va='center', fontsize=16, color='black')
ax.text(6, 19.5, 'EMO', ha='center', va='center', fontsize=16, color='black')
ax.text(8, 19.5, 'ENX', ha='center', va='center', fontsize=16, color='black')
ax.text(10, 19.5, 'MV2', ha='center', va='center', fontsize=16, color='black')
ax.text(12, 19.5, 'MV', ha='center', va='center', fontsize=16, color='black')
ax.text(14, 19.5, 'LVT', ha='center', va='center', fontsize=16, color='black')

# 添加图例
legend_patches = [
    patches.Patch(color=color, label=label)
    for label, color in color_map.items() if label != 'empty'
]
ax.legend(handles=legend_patches, loc='upper center', ncol=6, fontsize=16, bbox_to_anchor=(0.5, 1.1), frameon=False)

# 调整坐标范围和网格显示
ax.set_xlim(0, n_cols)
ax.set_ylim(0, n_rows)


# 隐藏坐标轴的刻度线和标签
ax.set_xticks([])  # 移除 X 轴刻度
ax.set_yticks([])  # 移除 Y 轴刻度
ax.tick_params(axis='both', which='both', length=0)  # 移除所有刻度线

plt.tight_layout()
plt.show()

#保存图片
output_folder = "pdf"
output_file = "pdf/fig12a_CpuBestEngine.pdf"
os.makedirs(output_folder, exist_ok=True)  # 确保文件夹存在
plt.savefig(output_file, format="pdf", bbox_inches="tight")
print(f"图片已保存到 {output_file}")