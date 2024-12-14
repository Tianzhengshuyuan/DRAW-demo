import json
import random
import os
import numpy as np

from myplot import MyPlot

save_path = os.path.join("pdf", "fp16vsint8_gpu.pdf") 


fig_cfg = {
    'type': 'stackbar',

    'categories': ['TensorRT FP16', 'TensorRT INT8'],
    'Latency':     [[0.9862, 0.7409, 0.3538, 1.8048],
                    [1.2417, 0.8529, 0.3731, 1.8593]],
    
    'color1': ['#434b95', '#1979c3', '#1cb3d0', '#33b636'],

    'color2': ['#b2311a', '#e06228', '#e38f37', '#e6b92a'],


    'label1': ['Conv_GEMM', 'Attn_MatMul', 'Attn_SoftMax', 'MISC_PWOP'],

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
