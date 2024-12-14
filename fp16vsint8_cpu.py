import json
import random
import os
import numpy as np

from myplot import MyPlot

save_path = os.path.join("pdf", "fp16vsint8_cpu.pdf") 


fig_cfg = {
    'type': 'groupstackbar',

    'categories': ['TFLite FP16', 'TFLite INT8', 'MNN FP16', 'MNN INT8'],
    'Latency':     [[11.836, 3.283, 0.809, 1.8],
                    [11.435, 3.246, 1.559, 2.632],
                    [11.722344, 1.841713, 2.139293, 1.195969,],  
                    [8.42308, 3.389613, 10.357842, 5.306907]],
    'Instruction': [[66221414, 7430213, 19527725, 1885217],  
                    [53500714, 21388533, 18687774, 4752851],
                    [59417379, 7956068,  8457857, 2299143],  
                    [70627467, 25591477, 16175182, 5453768]],
    # 'color1': ['#174ea6', '#4285f4', '#7baaf7', '#cfe2fa'],
    # 'color2': ['#a32620', '#ea4335', '#f77c6e', '#fbcfc7'],
    
    'color1': ['#434b95', '#1979c3', '#1cb3d0', '#33b636'],
    # 'color1': ['#434b95', '#7baaf7', '#1790a4', '#29902b'],

    # 'color2': ['#fc4a26', '#fb7832', '#f6ae42', '#facf33'],
    # 'color2': ['#c63b20', '#c95f28', '#d38a37', '#d1a52b'],
    'color2': ['#b2311a', '#e06228', '#e38f37', '#e6b92a'],

    'length': 10,
    'width': 6,

    
    'label1': ['Conv_GEMM', 'Attn_MatMul', 'Attn_SoftMax', 'MISC_PWOP'],
    'label2': ['simd', 'integer', 'load', 'store'],
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
