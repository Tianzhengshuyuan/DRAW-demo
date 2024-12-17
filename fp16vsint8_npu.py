import json
import random
import os
import numpy as np

from myplot import MyPlot

save_path = os.path.join("pdf", "fp16vsint8_npu.pdf") 


fig_cfg = {
    'type': 'stackbar',

    'categories': ['Openvino \n FP16', 'Openvino \n INT8', 'CANN FP16', 'CANN INT8'],
    'Latency':     [[0.403, 3.403, 2.577, 1.573, 0],
                    [0.734, 3.676, 4.022, 2.529, 0.686],
                    [6.9927, 0.4326, 3.1793, 10.6493, 0],
                    [7.1391, 1.2427, 3.1785, 11.8563, 0]],
    
    'color1': ['#434b95', '#1979c3', '#1cb3d0', '#33b636', '#b3e580'],

    'color2': ['#b2311a', '#e06228', '#e38f37', '#e6b92a'],

    'length': 5,
    'width': 6,
    'bar_width': 0.7,

    'label1': ['Conv_GEMM', 'Attn_MatMul', 'Attn_SoftMax', 'MISC_PWOP', 'Fake_Quant'],

    'pre_main_hook': lambda: print("I am a hook before main"),
    'post_main_hook': lambda: print("I am a hook after main"),

    # Misc
    'tight': True,

    # Save
    'save_path': save_path
}

if __name__ == '__main__':
    plot = MyPlot(fig_cfg)
    plot.plot()
