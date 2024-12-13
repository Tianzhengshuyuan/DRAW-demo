import os
import json
import random
import argparse
import numpy as np
from myplot import MyPlot


# 命令行参数解析
parser = argparse.ArgumentParser(description="Generate plots for specific model data")
parser.add_argument('--model', type=str, help="The model name (e.g., 'model1')")
parser.add_argument('--ymin', type=float, default=75, help="Minimum value for Y-axis (default: 75)")
parser.add_argument('--ymax', type=float, default=82, help="Maximum value for Y-axis (default: 82)")

# 解析命令行参数
args = parser.parse_args()
model_name = args.model

# 动态加载 JSON 数据和设置保存路径
data_file = os.path.join("json", f"{model_name}.json")   # 动态设置 JSON 文件
save_path = os.path.join("pdf", f"{model_name}.pdf") 

data = json.load(open(data_file))

fig_cfg = {
    'type': 'groupbar',

    'ylabel': 'Top-1 accuracy (%)',
    'ylabel_kwargs': {
        'fontsize': 15,
    },
    
    'ymin': args.ymin,
    'ymax': args.ymax,
    'categories': [datum['category'] for datum in data],
    'top1': [datum['top1'] for datum in data],
    'top1_div20': [datum['top1_div20'] for datum in data],
    'top1_div50': [datum['top1_div50'] for datum in data],

    # Misc
    'tight': True,
    'grid': True,
    'grid_linestyle': '--',
    'axis': 'y', #表示只显示grid中的横线，如果要只显示竖线的话设置为'x'

    # Save
    'save_path': save_path
}

if __name__ == '__main__':
    plot = MyPlot(fig_cfg)
    plot.plot()
