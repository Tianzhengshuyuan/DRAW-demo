import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# 数据示例：矩阵内容定义（每个单元格对应的标签）
data = [['GPU', 'EF2', 'SWF', 'EMO', 'ENX', 'MV2', 'MV', 'LVT'],
    ['G31', 'TF', 'TF', 'TF', 'TF', 'TF', 'TF', 'TF'],
    ['G52', 'TF', 'TF', 'MNN', 'TF', 'TF', 'TF', 'TF'],
    ['G610', 'TF', 'TF', 'MNN', 'TF', 'TF', 'TF', 'TF'],
    ['G77', 'TF', 'TF', 'MNN', 'TF', 'TF', 'TF', 'TF'],
    ['A630G', 'TF', 'TF', 'MNN', 'TF', 'NCNN', 'TF', 'TF'],
    ['A660G', 'TF', 'TF', 'MNN', 'TF', 'NCNN', 'TF', 'TF'],
    ['A740G', 'TF', 'TF', 'TF', 'TF', 'NCNN', 'TF', 'TF'],
    ['AMP', 'CoreML', 'CoreML', 'CoreML', 'CoreML', 'CoreML', 'CoreML', 'CoreML'],
    ['M1G', 'TensorRT', 'TensorRT', 'TensorRT', 'TensorRT', 'ONNX', 'TensorRT', 'TensorRT']
]


# 颜色映射
color_map = {
    'TF': '#5491f4',   # 蓝色
    'MNN': '#ec5548',      # 红色
    'NCNN': '#33a852',     # 绿色
    'ONNX': '#fabc04',   # 黄色
    'CoreML': '#D3D3D3',     # 灰色
    'TensorRT': '#46bdc5'     # 灰色
}

fig, ax = plt.subplots(figsize=(12, 6))

# 绘制网格
n_rows = len(data)
n_cols = len(data[0])

for i in range(n_rows):
    for j in range(n_cols):
        value = data[i][j]   # 空单元格填充灰色
        rect = patches.Rectangle(
            (j, n_rows - i - 1),  # 坐标 (x, y)
            1, 1,                 # 宽度和高度
            facecolor=color_map.get(value, '#FFFFFF'),  # 根据值填充颜色
            edgecolor='black'     # 设置边框颜色
        )
        if value not in color_map:
            ax.text(
                j + 0.5, n_rows - i - 0.5,  # 坐标中心
                value,
                ha='center', va='center', fontsize=20, color='black'
            )
        ax.add_patch(rect)


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
output_file = os.path.join(output_folder, "GpuBestEngine.pdf")
os.makedirs(output_folder, exist_ok=True)  # 确保文件夹存在
plt.savefig(output_file, format="pdf", bbox_inches="tight")
print(f"图片已保存到 {output_file}")