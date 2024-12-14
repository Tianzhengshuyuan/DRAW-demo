import json
import random
import os
import numpy as np

from myplot import MyPlot

data1_file = os.path.join("json", "linebar1.json")   # 动态设置 JSON 文件
data2_file = os.path.join("json", "linebar2.json")   # 动态设置 JSON 文件

data1 = json.load(open(data1_file))
data2 = json.load(open(data2_file))

save_path = os.path.join("pdf", "mylinebar.pdf") 

assert [datum['x'] for datum in data1] == [datum['x'] for datum in data2]
xs = [datum['x'] for datum in data1]

y1 = np.array([datum['y'] for datum in data1])
y2 = np.array([datum['y'] for datum in data2])



# r：红色（Red）# k：黑色（Black）
# y：黄色（Yellow）# g：绿色（Green）
def cat2color(cat):
    mapping = {
        'Stride': 'r',
        'Simple': 'k',
        'Complex': 'y',
        'Others': 'g',
    }

    for (k, v) in mapping.items():
        if k in cat:
            return v


def y2marker(datum):
    mapping = {
        'Berti': '*',
        'IPCP': '<',
        'SPP': '.',
        'MLOP': 'o',
    }

    for (k, v) in mapping.items():
        if k in datum['name']:
            return v


y1cat = ['Stride', "Simple", "Complex", "Others"]
y2cat = ['Stride', "Simple", "Complex", "Others"]

fig_cfg = {
    'type': 'linebar',
    #'title': 'test title',

    # X Data
    'x': xs,
    # X Label
    # 'xlabel': 'test x label',

    # X axis grouping
    'xgroup': True,
    'xgroup_kwargs': {
        'delimiter': '.',
        'minlevel': 1,
        'yfactor': 0.6,
        'yoffset': 0.2,
        'line_kwargs': {
            'lw': 0.7,
        },
        'text_kwargs': lambda lvl: {
            'rotation': 90 if lvl == 0 else 0
        },
    },

    'yaxes': [
        {
            'y': y1,
            'type': 'grouped_bar',
            'marker': '*',
            'color': [cat2color(cat) for cat in y1cat], #决定了bar和图例的颜色
            'label': y1cat,
            'axlabel': 'ylabel 1',  #决定左侧的label
            'grid': True,
            'grid_below': True,
            'grid_kwargs': {
                'linestyle': '--',
            },
            'legend': True,
            'legend_kwargs': {
                'frameon': False
            }
        },
        {
            'y': y2,
            'type': 'line',
            'linestyle': '--',  # '' to omit line
            'side': 'right', #决定label的位置是在图的右边
            'marker': 'o',
            # 'color': [cat2color(cat) for cat in y2cat],
            'color': 'b',
            'label': y2cat,
            'axlabel': 'ylabel 2',
        }
    ],

    'pre_main_hook': lambda: print("I am a hook before main"),
    'post_main_hook': lambda: print("I am a hook after main"),

    # Misc
    'tight': True,

    # Save
    'save_path': save_path
}



use_subplot_example = False
if use_subplot_example:
    subfig_cfg = fig_cfg
    # 创建一个cfg1，复制subfig_cfg的内容，并添加键值对 pos：(0,0)
    cfg1 = dict(subfig_cfg, pos=(0, 0))
    cfg2 = dict(subfig_cfg, pos=(1, 0))
    fig_cfg = {
        "figsize": [12, 6],
        "subplot": True,
        "gridspec_kwargs": {
            "nrows": 2,
            "ncols": 1,
            "hspace": 0,
            "wspace": 0,
        },
        "subplots_kwargs": {
            "sharex": True,
            "sharey": True,
        },
        "subplot_cfgs": [cfg1, cfg2],
        "tight": True,
    }

if __name__ == '__main__':
    plot = MyPlot(fig_cfg)
    plot.plot()
