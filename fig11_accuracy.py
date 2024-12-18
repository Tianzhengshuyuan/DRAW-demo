import os
import json
import random
import argparse
import numpy as np
from myplot import MyPlot

# 命令行参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str)
parser.add_argument('--ymin', type=float, default=75)
parser.add_argument('--ymax', type=float, default=82)

# 解析命令行参数
args = parser.parse_args()
input_name = args.input

# 动态加载 JSON 数据和设置保存路径
data_file = os.path.join("json", f"{input_name}.json")   # 动态设置 JSON 文件
save_path = os.path.join("pdf", f"{input_name}.pdf") 

data = json.load(open(data_file))

fig_cfg = {
    'type': 'groupbar_speedup_gpu',

    'ylabel': 'Speedup',
    'ylabel_kwargs': {
        'fontsize': 15,
    },
    
    'ymin': args.ymin,
    'ymax': args.ymax,
    'categories': [datum['category'] for datum in data],
    'G31': [datum['G31'] for datum in data],
    'G52': [datum['G52'] for datum in data],
    'G610': [datum['G610'] for datum in data],
    'G77': [datum['G77'] for datum in data],
    'A630G': [datum['A630G'] for datum in data],
    'A660G': [datum['A660G'] for datum in data],
    'A740G': [datum['A740G'] for datum in data],
    'AMP': [datum['AMP'] for datum in data],
    'MTL_GPU': [datum['MTL_GPU'] for datum in data],
    'LNL_GPU': [datum['LNL_GPU'] for datum in data],
    'ORIN_NPU': [datum['ORIN_NPU'] for datum in data],
    'AIP_NPU': [datum['AIP_NPU'] for datum in data],
    'MTL_NPU': [datum['MTL_NPU'] for datum in data],
    'LNL_NPU': [datum['LNL_NPU'] for datum in data],

    # Misc
    'tight': True,

    'axis': 'y', #表示只显示grid中的横线，如果要只显示竖线的话设置为'x'
    'figsize': [15,4],
    # Save
    'save_path': save_path
}

if __name__ == '__main__':
    plot = MyPlot(fig_cfg)
    plot.plot()
