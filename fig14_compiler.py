import os
import json
import random
import argparse
import numpy as np
from myplot import MyPlot


# 命令行参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--input_line', type=str, default="tflite_line")
parser.add_argument('--input_bar', type=str, default="tflite_bar")
parser.add_argument('--output', type=str, default="tflite_compiler_lib")
parser.add_argument('--ymin', type=float, default=0)
parser.add_argument('--ymax', type=float, default=9)

# 解析命令行参数
args = parser.parse_args()
line_file_name = args.input_line
bar_file_name = args.input_bar
output_file_name = args.output

# 动态加载 JSON 数据和设置保存路径
line_data_file = os.path.join("json", f"{line_file_name}.json")   # 动态设置 JSON 文件
bar_data_file = os.path.join("json", f"{bar_file_name}.json")   # 动态设置 JSON 文件
save_path = os.path.join("pdf", f"{output_file_name}.pdf") 

line_data = json.load(open(line_data_file))
bar_data = json.load(open(bar_data_file))

fig_cfg = {
    'type': 'grouplinebar',

    'ylabel': 'Latency ratio to TFLite',
    'ylabel_kwargs': {
        'fontsize': 15,
    },
    
    'ymin': args.ymin,
    'ymax': args.ymax,
    'categories': [datum['category'] for datum in line_data],
    'A55_bar': [datum['A55'] for datum in bar_data],
    'A76_bar': [datum['A76'] for datum in bar_data],
    'M1P_bar': [datum['M1P'] for datum in bar_data],
    'A55_line': [datum['A55'] for datum in line_data],
    'A76_line': [datum['A76'] for datum in line_data],
    'M1P_line': [datum['M1P'] for datum in line_data],
    # Misc
    'tight': True,

    'axis': 'y', #表示只显示grid中的横线，如果要只显示竖线的话设置为'x'
    'figsize': [12,5],
    # Save
    'save_path': save_path
}

if __name__ == '__main__':
    plot = MyPlot(fig_cfg)
    plot.plot()
