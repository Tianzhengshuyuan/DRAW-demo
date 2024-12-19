import os
import json
import random

import numpy as np
from myplot import MyPlot


# 动态加载 JSON 数据和设置保存路径
tflite_line_data_file = os.path.join("json", "tflite_line.json")  
tflite_bar_data_file = os.path.join("json", "tflite_bar.json")  
 
mnn_line_data_file = os.path.join("json", "mnn_line.json")  
mnn_bar_data_file = os.path.join("json", "mnn_bar.json") 

onnx_line_data_file = os.path.join("json", "onnx_line.json")  
onnx_bar_data_file = os.path.join("json", "onnx_bar.json") 

save_path = os.path.join("pdf", "Compiler_lib.pdf") 

tflite_line_data = json.load(open(tflite_line_data_file))
tflite_bar_data = json.load(open(tflite_bar_data_file))
mnn_line_data = json.load(open(mnn_line_data_file))
mnn_bar_data = json.load(open(mnn_bar_data_file))
onnx_line_data = json.load(open(onnx_line_data_file))
onnx_bar_data = json.load(open(onnx_bar_data_file))

subfig1_cfg = {
    'type': 'grouplinebar',
    'figsize': [12,5],
    'ylabel': 'Latency ratio to TFLite',
    'ylabel_kwargs': {
        'fontsize': 15,
    },
    
    'first': True,
    'last': False,
    
    'ymin': 0,
    'ymax': 9,
    
    'categories': [datum['category'] for datum in tflite_line_data],
    'A55_bar': [datum['A55'] for datum in tflite_bar_data],
    'A76_bar': [datum['A76'] for datum in tflite_bar_data],
    'M1P_bar': [datum['M1P'] for datum in tflite_bar_data],
    'A55_line': [datum['A55'] for datum in tflite_line_data],
    'A76_line': [datum['A76'] for datum in tflite_line_data],
    'M1P_line': [datum['M1P'] for datum in tflite_line_data]
}

subfig2_cfg = {
    'type': 'grouplinebar',
    'figsize': [12,5],
    'ylabel': 'Latency ratio to MNN',
    'ylabel_kwargs': {
        'fontsize': 15,
    },
    
    'first': False,
    'last': False,
    
    'ymin': 0,
    'ymax': 10,
    
    'categories': [datum['category'] for datum in mnn_line_data],
    'A55_bar': [datum['A55'] for datum in mnn_bar_data],
    'A76_bar': [datum['A76'] for datum in mnn_bar_data],
    'M1P_bar': [datum['M1P'] for datum in mnn_bar_data],
    'A55_line': [datum['A55'] for datum in mnn_line_data],
    'A76_line': [datum['A76'] for datum in mnn_line_data],
    'M1P_line': [datum['M1P'] for datum in mnn_line_data],
}

subfig3_cfg = {
    'type': 'grouplinebar',
    'figsize': [12,5],
    'ylabel': 'Latency ratio to ONNXRT',
    'ylabel_kwargs': {
        'fontsize': 15,
    },

    'first': False,
    'last': True,
    
    'ymin': 0,
    'ymax': 4,
    
    'categories': [datum['category'] for datum in onnx_line_data],
    'A55_bar': [datum['A55'] for datum in onnx_bar_data],
    'A76_bar': [datum['A76'] for datum in onnx_bar_data],
    'M1P_bar': [datum['M1P'] for datum in onnx_bar_data],
    'A55_line': [datum['A55'] for datum in onnx_line_data],
    'A76_line': [datum['A76'] for datum in onnx_line_data],
    'M1P_line': [datum['M1P'] for datum in onnx_line_data],
}

fig_cfg = {
    'type': 'grouplinebar',

    'ylabel': 'Latency ratio to TFLite',
    'ylabel_kwargs': {
        'fontsize': 15,
    },
    
    
    "subplot": True,
    'subplot_cfgs': [subfig1_cfg, subfig2_cfg, subfig3_cfg],
    # Misc
    'tight': True,

    'axis': 'y', #表示只显示grid中的横线，如果要只显示竖线的话设置为'x'
    'figsize': [12,15],
    # Save
    'save_path': save_path
}

if __name__ == '__main__':
    plot = MyPlot(fig_cfg)
    plot.plot()
