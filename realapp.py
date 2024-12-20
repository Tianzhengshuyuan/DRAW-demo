import os
import json
import random
import argparse
import numpy as np
from myplot import MyPlot


# 命令行参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default="timm-models-a")
parser.add_argument('--ymin', type=float, default=0)
parser.add_argument('--ymax', type=float, default=130)

# 解析命令行参数
args = parser.parse_args()
input_name = args.input

# 动态加载 JSON 数据和设置保存路径
data_file = os.path.join("json", f"{input_name}.json")   # 动态设置 JSON 文件
save_path = os.path.join("pdf", f"{input_name}.pdf") 

data = json.load(open(data_file))

fig_cfg = {
    'type': 'groupbar_app',

    'ylabel': '#models',
    'ylabel_kwargs': {
        'fontsize': 15,
    },
    
    'ymin': args.ymin,
    'ymax': args.ymax,
    
    'category': [datum['category'] for datum in data],
    'etbench': [datum['etbench'] for datum in data],
    'app': [datum['app'] for datum in data],

    # Misc
    'tight': True,
    # 'grid': True,
    # 'grid_linestyle': '--',
    'axis': 'y', #表示只显示grid中的横线，如果要只显示竖线的话设置为'x'

    'figsize' : [10,6],
    # Save
    'save_path': save_path
}

if __name__ == '__main__':
    plot = MyPlot(fig_cfg)
    plot.plot()
